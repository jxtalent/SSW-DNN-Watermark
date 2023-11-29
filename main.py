import argparse
from argparse import Namespace

import yaml

from train.classification import Classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Parameters for running model watermarking experiments")

    parser.add_argument('--action', default='watermark', choices=['clean', 'watermark', 'attack', 'evaluate'],
                        help='watermark: train a watermarked model,\n'
                             'clean: train a clean model,\n'
                             'attack: attack a given model,\n'
                             'evaluate: test a given model')

    parser.add_argument('--dataset', default='fashion', help='Dataset used to train a model (default: fashion)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--data_path', default='./data', help='Path to store all the relevant datasets (default: '
                                                              './data)')

    parser.add_argument('--arch', default='vanilla',
                        choices=['vanilla', 'resnet18', 'resnet34', 'resnet50', 'simple', 'mobile', 'VGG16', 'senet'],
                        help='Model architecture (default: vanilla)\n')

    parser.add_argument('--epochs', type=int, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    parser.add_argument('--gpu', default='0', type=str, help='GPU index')
    parser.add_argument('--seed', default=None, type=int, help='Seed for random')

    parser.add_argument('--log_dir', default='logs', help='Logging directory')
    parser.add_argument('--runname', default='', help='Name for an experiment')
    parser.add_argument('--save-interval', type=int, default=100, help='Save model interval')

    # evaluate a model, or resume from a model
    parser.add_argument('--checkpoint_path', help='Checkpoint path (example: xxx/xxx/last.pt)')

    # watermark
    parser.add_argument('--key_num', default=100, type=int, help='Trigger set size in training')
    parser.add_argument('--key_target', default=1, type=int, help='Target label')

    # watermark a pretrained model
    parser.add_argument('--clean_model_path', type=str, default=None, help='Path to the non-watermarked model')
    parser.add_argument('--from_pretrained', type=bool, default=True, help='Path to the non-watermarked model')
    parser.add_argument('--wm_lr', default=0.1, type=float, help='Learning rate for the trigger images')
    parser.add_argument('--wm_epoch', default=5, type=int, help='Epochs to optimize the trigger images')
    parser.add_argument('--save_trigger', action='store_true', help='Save the trigger set as png images')
    parser.add_argument('--wm_it', default=20, type=int, help='Iterations to optimize the trigger images')

    # attack a model
    parser.add_argument('--victim_path', default=None, help='Victim model path (example: xxx/xxx/last.pt)')
    parser.add_argument('--extract_soft', default=0, type=int, help='Model extraction with probability vectors')

    # retrain means hard label extraction
    parser.add_argument('--attack_type', default=None, choices=['retrain', 'distill', 'knockoff', 'cross',
                                                                'ftal', 'rtal', 'prune', 'quantization'],
                        type=str, help='Attack type')

    parser.add_argument('--config', type=str, default=None, help='Path to YAML configuration file')

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as config_file:
            config_args = yaml.safe_load(config_file)
            for key, value in config_args.items():
                setattr(args, key, value)

    return args


if __name__ == '__main__':
    args = parse_args()

    classifier = Classifier(args)
    if args.action != 'evaluate':
        # action in ['clean', 'watermark', 'attack']
        classifier.train()
    else:
        classifier.evaluate()
