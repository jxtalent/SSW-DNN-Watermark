import torch
import time
import torch.nn.functional as functional
from tqdm import tqdm


def train(net, criterion, optimizer, trainloader, device, has_victim=False, victim=None, logits=False, T=1.0, alpha=1.0):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        if has_victim:
            victim.eval()
            victim_outputs = victim(inputs).detach()
            victim_targets = victim_outputs.max(1)[1]
            if logits:
                # victim's logit is accessible
                loss = alpha * (T ** 2) * functional.kl_div(functional.log_softmax(outputs / T, dim=1),
                                                            functional.softmax(victim_outputs / T, dim=1),
                                                            reduction='batchmean') + (1 - alpha) * criterion(outputs, targets)
            else:
                loss = criterion(outputs, victim_targets)
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {'loss': train_loss / len(trainloader),
            'acc': correct / total,
            'time': time.time() - start_time}


def train_by_soft_label(net, optimizer, trainloader, device, use_kl=False):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        if use_kl:
            loss = functional.kl_div(functional.log_softmax(outputs, dim=1), targets, reduction='batchmean')
        else:
            loss = - (torch.log_softmax(outputs, 1) * targets).sum(1).mean()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets.max(1)[1]).sum().item()

    return {'loss': train_loss / len(trainloader),
            'acc': correct / total,
            'time': time.time() - start_time}


def optimize_key(net, shadow, clean, criterion, key, k_optimizer, iterations=50):
    net.eval()
    shadow.eval()
    clean.eval()
    key.train()

    idx = 0
    key_loss = 0.
    batch_size = 100
    while idx + batch_size <= len(key.images):
        last_acc_clean = 100
        last_acc_shadow = 0
        it = 0
        while True:
            xt, targets = key.images[idx:idx + batch_size], key.targets[idx:idx + batch_size]
            k_optimizer.zero_grad()

            y_1 = net(xt)
            y_2 = shadow(xt)
            y_3 = clean(xt)
            T = 1.0
            fusion = torch.softmax(y_1 / T, -1) * torch.softmax(y_2 / T, -1) * (1 - torch.softmax(y_3 / T, -1))
            loss = criterion(fusion, targets)
            loss.backward()
            k_optimizer.step()

            key.images.data.clip_(0., 1.)

            it += 1

            acc_clean = validate_watermark(clean, key.images[idx:idx + batch_size], key.targets[idx:idx + batch_size])
            acc_shadow = validate_watermark(shadow, key.images[idx:idx + batch_size], key.targets[idx:idx + batch_size])

            # stop updating
            if last_acc_clean - acc_clean < 0.001 and acc_shadow - last_acc_shadow < 0.001 and it >= iterations:
                key_loss += loss.item()
                break

            last_acc_clean = acc_clean
            last_acc_shadow = acc_shadow

        idx += batch_size

    return key_loss / (len(key.images) / batch_size)


def train_on_watermark(net, criterion, optimizer, trainloader, key, device, k=12):
    net.train()
    key.eval()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    wmidx = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        xt = key.images[wmidx: wmidx + k]
        yt = key.targets[wmidx: wmidx + k]
        wmidx += k
        if wmidx + k > len(key.images):
            wmidx = 0

        outputs = net(torch.cat([inputs, xt], 0))
        targets = torch.cat([targets, yt], 0)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {'loss': train_loss / len(trainloader),
            'acc': correct / total,
            'time': time.time() - start_time}


def test(net, criterion, testloader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return {'loss': test_loss / len(testloader),
            'acc': (correct / total),
            'time': time.time() - start_time}


def test_on_watermark(net, criterion, key):
    net.eval()
    key.eval()
    test_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    predictions = []

    with torch.no_grad():
        xt, targets = key.images, key.targets
        batches = len(xt) // 100 + int(len(xt) % 100 > 0)
        for batch_idx in range(batches):
            curr_inputs = xt[batch_idx * 100: min((batch_idx + 1) * 100, len(xt))]
            curr_targets = targets[batch_idx * 100: min((batch_idx + 1) * 100, len(xt))]
            outputs = net(curr_inputs)
            loss = criterion(outputs, curr_targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += curr_targets.size(0)
            correct += predicted.eq(curr_targets).sum().item()
            predictions.append(predicted)

    return {'loss': test_loss / batches,
            'acc': (correct / total),
            'time': time.time() - start_time,
            'prediction': torch.cat(predictions)}


def validate_watermark(net, trigger, target):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        xt = trigger
        for batch_idx in range(len(xt) // 100 + int(len(xt) % 100 > 0)):
            inputs = xt[batch_idx * 100: min((batch_idx + 1) * 100, len(xt))]
            outputs = net(inputs)
            predicted = outputs.max(1)[1]
            correct += predicted.eq(target).sum().item()
            total += inputs.size(0)
    return correct / total
