import os.path
import random
from pprint import pprint
import sys

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision

from models.resnet import ResNet18, ResNet34, ResNet50, NaiveCNN, MoreNaiveCNN
from models.mobilenetv2 import MobileNetV2
from models.vgg import VGG
from models.watermark import Key
from models.senet import SENet18
from models.resnet_imagenet import imagenet_get_model
import torch.backends.cudnn as cudnn
import json

from .attack import weight_prune, quantization, re_initializer_layer
from .utils import append_history, Logger
from .trainer import *
from datasets.dataloader import *


class Classifier:
    def __init__(self, args):
        self.args = args
        self._set_device(args)
        if args.action != 'evaluate':
            self._set_log()
        self.prepare_dataset()

        self.net = self.build_model()
        self.optimizer, self.scheduler = self.build_optimizer(self.net)

    def train(self):
        self.start = time.time()
        self._train_model()
        end = time.time()
        print('Total time:%.4f min' % ((end - self.start) / 60))

    def evaluate(self):
        assert self.args.checkpoint_path is not None, "To test a model, provide the model path"

        checkpoint = torch.load(self.args.checkpoint_path)
        self.net.load_state_dict(checkpoint['net'])
        self._evaluate()

        dirname = os.path.dirname(self.args.checkpoint_path)
        trigger_path = os.path.join(dirname, 'key.pt')
        if os.path.exists(trigger_path):
            key = torch.load(trigger_path)
            n = 100
            key.images.data = key.images.data[:n]
            key.targets = key.targets[:n]  # use how many triggers to test the accuracy
            self._evaluate_on_trigger(key)

    def prepare_dataset(self):
        self.trainloader, self.testloader, self.num_classes = get_dataloader(self.args.dataset, self.args.batch_size,
                                                                             data_path=self.args.data_path)

    def build_model(self):
        if 'VGG' in self.args.arch.upper():
            assert self.args.arch.upper() in ['VGG11', 'VGG13', 'VGG16', 'VGG19']
            net = VGG(self.args.arch.upper(), self.num_classes).to(self.device)
        elif 'senet' in self.args.arch:
            net = SENet18(self.num_classes).to(self.device)
        elif 'imgnet' in self.args.arch:
            net = imagenet_get_model('res34').to(self.device)
        else:
            Network = {'resnet18': ResNet18,
                       'resnet34': ResNet34,
                       'resnet50': ResNet50,
                       'vanilla': NaiveCNN,
                       'simple': MoreNaiveCNN,
                       'mobile': MobileNetV2}[self.args.arch]
            net = Network(self.num_classes).to(self.device)
        return net

    def build_optimizer(self, net):
        if self.args.dataset in ['cifar10', 'cifar100', 'fashion', 'mnist']:
            optimizer = optim.Adam(net.parameters(), lr=self.args.lr, betas=(0.9, 0.99), eps=1e-5)
            scheduler = None

        else:
            optimizer = optim.SGD(net.parameters(), lr=self.args.lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        return optimizer, scheduler

    def _train_model(self):
        if self.args.action == 'clean':
            self._train_clean()
        elif self.args.action == 'watermark':
            self._train_watermark()
        else:
            self._attack()

    def _train_clean(self):
        criterion = nn.CrossEntropyLoss()
        start_epoch = 0
        best_acc = 0.

        if self.args.checkpoint_path is not None:
            checkpoint = torch.load(self.args.checkpoint_path)
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1
            self.net.load_state_dict(checkpoint['net'])

        for epoch in range(start_epoch, self.args.epochs):
            print('\nEpoch: %d' % epoch)
            train_metrics = train(self.net, criterion, self.optimizer, self.trainloader, self.device)

            valid_metrics = self._evaluate()

            if self.scheduler is not None:
                self.scheduler.step()

            metrics = {'epoch': epoch}
            for k in train_metrics: metrics[f'train_{k}'] = train_metrics[k]
            for k in valid_metrics: metrics[f'valid_{k}'] = valid_metrics[k]

            best_acc = self._log_metrics_and_models(metrics, epoch == start_epoch, epoch, best_acc)

    def _build_trigger_set(self):
        channels = {'cifar10': 3, 'cifar100': 3, 'fashion': 1, 'mnist': 1}[self.args.dataset]
        key = Key(self.args.key_num, self.args.key_target, channels, self.device, self.args.data_path).to(self.device)
        return key

    def _train_watermark(self):
        positive_shadow = self.build_model()
        shadow_optimizer, shadow_scheduler = self.build_optimizer(positive_shadow)

        negative_shadow = self.build_model()
        assert self.args.clean_model_path is not None, "Provide a clean model as the negative shadow model"
        negative_shadow.load_state_dict(torch.load(self.args.clean_model_path)['net'])

        key = self._build_trigger_set()
        k_optimizer = optim.Adam(key.parameters(), lr=self.args.wm_lr)

        if self.args.from_pretrained:
            self.net.load_state_dict(torch.load(self.args.clean_model_path)['net'])
            positive_shadow.load_state_dict(torch.load(self.args.clean_model_path)['net'])

        # dataloader, dataset is not augmented
        dataloader, _, _ = get_dataloader(self.args.dataset, self.args.batch_size, augment=False)
        # dataloader for training the shadow model, dataset is augmented
        shadow_trainloader, _, _ = get_dataloader(self.args.dataset, self.args.batch_size)

        criterion = nn.CrossEntropyLoss()
        start_epoch = 0
        best_acc = 0.

        for epoch in range(start_epoch, self.args.epochs):
            print('\nEpoch: %d' % epoch)

            # train the host model
            train_metrics = train_on_watermark(self.net, criterion, self.optimizer, self.trainloader, key, self.device)
            if self.scheduler:
                self.scheduler.step()

            # use the host model's prediction to label the dataset
            self.net.eval()
            labels = [self.net(inputs.to(self.device)).max(1)[1].cpu() for inputs, _ in dataloader]
            if self.args.dataset == 'fashion':
                shadow_trainloader.dataset.targets = torch.cat(labels)
            else:
                shadow_trainloader.dataset.targets = [t.item() for t in torch.cat(labels)]

            # train the positive shadow model
            train_metrics_s = train(positive_shadow, criterion, shadow_optimizer, shadow_trainloader, self.device)
            if shadow_scheduler:
                shadow_scheduler.step()

            valid_metrics = self._evaluate(verbose=False)
            valid_metrics_pos = self._evaluate(model=positive_shadow, verbose=False)
            print(f"Test acc | Host net: {valid_metrics['acc']:.4f}")
            print(f"Test acc | Positive shadow net: {valid_metrics_pos['acc']:.4f}")

            wm_metrics = self._evaluate_on_trigger(key, model=self.net, verbose=False)
            print(f"Trigger set acc | Host net: {wm_metrics['acc']:.4f}")

            # update the trigger set
            if epoch >= self.args.epochs - self.args.wm_epoch:
                key_loss = optimize_key(self.net, positive_shadow, negative_shadow, criterion, key, k_optimizer, self.args.wm_it)
                wm_metrics_pos = self._evaluate_on_trigger(key, model=positive_shadow, verbose=False)
                wm_metrics_neg = self._evaluate_on_trigger(key, model=negative_shadow, verbose=False)
                print(f"Trigger set acc | Positive shadow net: {wm_metrics_pos['acc']:.4f}")
                print(f"Trigger set acc | Negative shadow net: {wm_metrics_neg['acc']:.4f}")

            metrics = {'epoch': epoch}
            for k in train_metrics: metrics[f'train_{k}'] = train_metrics[k]
            for k in valid_metrics: metrics[f'valid_{k}'] = valid_metrics[k]
            for k in wm_metrics: metrics[f'wm_{k}'] = wm_metrics[k]

            best_acc = self._log_metrics_and_models(metrics, epoch == start_epoch, epoch, best_acc)

        # Trigger selection
        prediction_on_trigger = test_on_watermark(self.net, criterion, key)['prediction']
        select = prediction_on_trigger.eq(key.targets)

        prediction_on_trigger = test_on_watermark(positive_shadow, criterion, key)['prediction']
        select = prediction_on_trigger.eq(key.targets) & select

        prediction_on_trigger = test_on_watermark(negative_shadow, criterion, key)['prediction']
        select = prediction_on_trigger.not_equal(key.targets) & select

        print('Selected trigger set size', select.sum().item())

        key.images.data = key.images.data[select]
        key.targets = key.targets[select]
        torch.save(key, os.path.join(self.log_dir, 'key.pt'))

        if self.args.save_trigger:
            os.makedirs(os.path.join(self.log_dir, 'trigger', 'examples'), exist_ok=True)
            # grid visualization
            torchvision.utils.save_image(key.images[:20], os.path.join(self.log_dir, 'trigger', 'grid.png'),
                                         normalize=True, nrow=5)
            # individual examples
            for i, image in enumerate(key.images.data):
                img = torch.clamp(image, min=0.0, max=1.0)
                img = img.permute(1, 2, 0).cpu().numpy()
                img = np.squeeze(img)
                img = (img * 255.).astype(np.uint8)
                img = Image.fromarray(img)
                img.save(os.path.join(self.log_dir, 'trigger', 'examples', f'{i:04}.png'))

    def _attack(self):
        """attack a watermarked model"""
        # model extraction attack / model modification attack
        assert self.args.victim_path is not None, "To attack a model, provide the victim model path"
        if self.args.dataset == 'fashion':
            victim = NaiveCNN(10).to(self.device)
        elif self.args.dataset == 'cifar10':
            victim = ResNet18(10).to(self.device)
        else:
            victim = ResNet18(100).to(self.device)
        victim.load_state_dict(torch.load(self.args.victim_path)['net'])
        self.key = torch.load(os.path.join(os.path.dirname(self.args.victim_path), 'key.pt')).to(self.device)

        valid_metrics = self._evaluate(model=victim, verbose=False)
        print(f"Test acc | Victim net: {valid_metrics['acc']:.4f}")
        wm_metrics = self._evaluate_on_trigger(self.key, model=victim, verbose=False)
        print(f"Trigger set acc | Victim net: {wm_metrics['acc']:.4f}")

        if self.args.attack_type == 'prune':
            self.__prune_attack(victim)
        elif self.args.attack_type == 'quantization':
            self.__weight_quantization(victim)
        elif self.args.attack_type in ['ftll', 'ftal', 'rtll', 'rtal']:
            self.__fine_tune(victim)
        else:
            self.__model_extract(victim)

    def _set_device(self, args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if args.seed is None:
            args.seed = random.randint(0, 2023)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.benchmark = True

    def _set_log(self):
        if self.args.action != 'attack':
            # path to a clean / watermarked model
            log_dir = os.path.join(self.args.log_dir, self.args.dataset, self.args.action)
            log_dir = os.path.join(log_dir, time.strftime("%m-%d-%H-%M-", time.localtime()) + self.args.runname)
        else:
            # path to the attacked model
            log_dir = os.path.split(self.args.victim_path)[0]  # victim model's dir
            log_dir = os.path.join(log_dir, 'attack', self.args.attack_type, self.args.runname)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, 'log.txt')
        self.config_file = os.path.join(log_dir, 'conf.json')
        self.history_file = os.path.join(log_dir, 'history.csv')

        json.dump(vars(self.args), open(self.config_file, 'w'), indent=4)
        pprint(vars(self.args))

        sys.stdout = Logger(filename=self.log_file, stream=sys.stdout)

    def _evaluate(self, model=None, verbose=True):
        criterion = nn.CrossEntropyLoss()
        if model is None:
            model = self.net
        valid_metrics = test(model, criterion, self.testloader, self.device)
        if verbose:
            print('Test acc: %.4f, test loss: %.4f' % (valid_metrics['acc'], valid_metrics['loss']))
        return valid_metrics

    def _evaluate_on_trigger(self, key, model=None, verbose=True, return_pred=False):
        criterion = nn.CrossEntropyLoss()
        if model is None:
            model = self.net
        valid_metrics = test_on_watermark(model, criterion, key)
        if verbose:
            print('Trigger set acc: %.4f' % (valid_metrics['acc']))
        if not return_pred:
            del valid_metrics['prediction']
        return valid_metrics

    def _log_metrics_and_models(self, metrics, firstrow, epoch, best_acc):
        append_history(self.history_file, metrics, firstrow)

        state = {
            'net': self.net.state_dict(),
            'acc': metrics['valid_acc'],
            'epoch': epoch
        }

        if self.args.save_interval and (epoch + 1) % self.args.save_interval == 0:
            torch.save(state, os.path.join(self.log_dir, f'epoch-{epoch}.pth'))

        if best_acc < metrics['valid_acc']:
            print(f'Found best at epoch {epoch}\n')
            best_acc = metrics['valid_acc']
            torch.save(state, os.path.join(self.log_dir, 'best.pt'))

        torch.save(state, os.path.join(self.log_dir, 'last.pt'))

        return best_acc

    def __model_extract(self, victim):
        device = self.device
        key = self.key

        criterion = nn.CrossEntropyLoss()
        start_epoch = 0
        best_acc = 0.

        # cross structure retraining (hard label extraction, using a different structure from the host model)
        if self.args.attack_type == 'cross':
            self.net = self.build_model()
            self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 40], gamma=0.1)

        # distilling, using soft label, using a smaller architecture
        if self.args.attack_type == 'distill':
            self.args.extract_soft = 1
            self.args.arch = {'cifar10': 'mobile', 'cifar100': 'mobile', 'fashion': 'simple'}[self.args.dataset]
            self.net = self.build_model()
            self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-5)

        # knockoff nets
        use_transfer = True if self.args.attack_type == 'knockoff' else False
        transfer_set = {'cifar10': 'cifar100', 'cifar100': 'imagenet32', 'fashion': 'graycifar10'}[self.args.dataset]

        # dataset used by the attacker
        extracted_loader, _, _ = get_dataloader(transfer_set if use_transfer else self.args.dataset,
                                                self.args.batch_size, augment=False)
        print("Attacker's training set size:", len(extracted_loader.dataset))
        victim.eval()

        if self.args.extract_soft or use_transfer:
            # using probability
            with torch.no_grad():
                extracted_prob = [torch.softmax(victim(inputs.to(device)), dim=1).data.cpu() for inputs, _ in
                                  extracted_loader]  # each item: torch.size([bs, #classes])
                soft_label = torch.cat(extracted_prob, dim=0)  # (num_samples, #classes)
            extracted_loader, _, _ = get_soft_label_dataloader(soft_label,
                                                               transfer_set if use_transfer else self.args.dataset)
        else:
            # using hard label
            with torch.no_grad():
                extracted_label = [victim(inputs.to(device)).max(1)[1].data.cpu() for inputs, _ in
                                   extracted_loader]  # each item: torch.size([bs,])
                extracted_label = torch.cat(extracted_label)

            extracted_loader, _, _ = get_dataloader(transfer_set if use_transfer else self.args.dataset,
                                                    self.args.batch_size)

            extracted_loader.dataset.targets = [t.item() for t in extracted_label]

        for epoch in range(start_epoch, self.args.epochs):
            print('\nEpoch: %d' % epoch)
            if self.args.extract_soft or use_transfer:
                train_metrics = train_by_soft_label(self.net, self.optimizer, extracted_loader, device,
                                                    self.args.attack_type == 'distill')
            else:
                train_metrics = train(self.net, criterion, self.optimizer, extracted_loader, self.device)

            valid_metrics = self._evaluate(verbose=False)
            print(f"Test acc | Surrogate net: {valid_metrics['acc']:.4f}")
            wm_metrics = self._evaluate_on_trigger(self.key, verbose=False)
            print(f"Trigger set acc | Surrogate net: {wm_metrics['acc']:.4f}")

            if self.scheduler is not None:
                self.scheduler.step()

            metrics = {'epoch': epoch}
            for k in train_metrics: metrics[f'train_{k}'] = train_metrics[k]
            for k in valid_metrics: metrics[f'valid_{k}'] = valid_metrics[k]
            for k in wm_metrics: metrics[f'wm_{k}'] = wm_metrics[k]

            best_acc = self._log_metrics_and_models(metrics, epoch == start_epoch, epoch, best_acc)

    def __prune_attack(self, victim):
        key = self.key

        print('Pruning the model...')
        for perc in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            print(f'\nPruning rate: {perc}%')
            pruned_model = weight_prune(victim, perc)
            metrics = self._evaluate(model=pruned_model)
            wm_metrics = self._evaluate_on_trigger(key, model=pruned_model)

            # save the pruned model
            if perc == 60:
                state = {
                    'net': pruned_model.state_dict(),  # save state_dict()
                    'acc': metrics['acc'],
                    'epoch': 0,
                }
                torch.save(state, os.path.join(self.log_dir, 'last.pt'))

    def __weight_quantization(self, victim, bits=3):
        key = self.key

        for name, param in victim.named_parameters():
            quantization(param, bits=bits)

        print(f'Quantifying the model with {bits} bits...')
        test_acc = self._evaluate(model=victim, verbose=True)['acc']
        wm_acc = self._evaluate_on_trigger(key, model=victim, verbose=True)['acc']
        state = {
            'net': victim.state_dict(),
            'acc': test_acc,
            'epoch': 0,
        }
        torch.save(state, os.path.join(self.log_dir, 'last.pt'))

    def __fine_tune(self, victim):
        self.net = victim
        key = self.key

        training_set_size = len(self.trainloader.dataset)
        perc = 0.2
        select_index = random.sample(list(range(training_set_size)), int(training_set_size * perc))
        self.trainloader.dataset.data = self.trainloader.dataset.data[select_index]
        if self.args.dataset == 'fashion':
            self.trainloader.dataset.targets = self.trainloader.dataset.targets[select_index]
        else:
            tgs = self.trainloader.dataset.targets
            self.trainloader.dataset.targets = list(np.array(tgs)[select_index])
        print(f'Attacker training set size:{len(self.trainloader.dataset)}')

        if self.args.attack_type == 'rtal':
            self.net, original_last_layer = re_initializer_layer(self.net, self.num_classes, self.device)

        start_epoch = 0
        best_acc = 0.
        criterion = nn.CrossEntropyLoss()

        for epoch in range(start_epoch, self.args.epochs):
            print('\nEpoch: %d' % epoch)
            train_metrics = train(self.net, criterion, self.optimizer, self.trainloader, self.device)

            if self.scheduler is not None:
                self.scheduler.step()

            valid_metrics = self._evaluate(verbose=False)
            print('Test acc: %.4f, test loss: %.4f' % (valid_metrics['acc'], valid_metrics['loss']))

            if self.args.attack_type == 'rtal':
                self.net, new_layer = re_initializer_layer(self.net, self.num_classes, self.device, original_last_layer)
            wm_metrics = self._evaluate_on_trigger(key, verbose=False)
            print(f"Watermark acc: {wm_metrics['acc']:.6f}")

            metrics = {'epoch': epoch}
            for k in train_metrics: metrics[f'train_{k}'] = train_metrics[k]
            for k in valid_metrics: metrics[f'valid_{k}'] = valid_metrics[k]
            for k in valid_metrics: metrics[f'watermark_{k}'] = wm_metrics[k]

            best_acc = self._log_metrics_and_models(metrics, epoch == start_epoch, epoch, best_acc)

            if self.args.attack_type == 'rtal':
                self.net, _ = re_initializer_layer(self.net, self.num_classes, self.device, new_layer)
