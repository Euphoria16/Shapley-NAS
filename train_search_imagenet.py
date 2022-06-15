import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import copy

from torch.autograd import Variable
from model_search_imagenet import Network

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--data', type=str, default='/tmp/cache/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--resume', type=str, default='', help='resume from pretrained')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='/tmp/checkpoints/', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--shapley_momentum', type=float, default=0.8, help='momentum for updating shapley')
parser.add_argument('--step_size', type=float, default=0.1, help='step size for updating shapley')
parser.add_argument('--samples', type=int, default=10, help='number of samples for estimation')
parser.add_argument('--threshold', type=float, default=0.5, help='early truncation threshold')

parser.add_argument('--begin', type=int, default=10, help='batch size')

parser.add_argument('--tmp_data_dir', type=str, default='/cache/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 1000


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)

    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    # dataset split
    train_data1 = dset.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    train_data2 = dset.ImageFolder(valdir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    valid_data = dset.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    num_train = len(train_data1)
    num_val = len(train_data2)
    print('# images to train network: %d' % num_train)
    print('# images to validate network: %d' % num_val)

    model = Network(args.init_channels, CLASSES, args.layers, criterion)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    start_epoch = 0

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    optimizer_a = torch.optim.Adam(model.module.arch_parameters(),
                                   lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                   weight_decay=args.arch_weight_decay)

    train_queue = torch.utils.data.DataLoader(
        train_data1, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data2, batch_size=1024, shuffle=True,
        pin_memory=True, num_workers=args.workers)

    infer_queue = torch.utils.data.DataLoader(
        train_data2, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.workers)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    if args.resume:
        checkpoint = torch.load(os.path.join(args.resume, 'checkpoint.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])

        model.module.show_arch_parameters()
        genotype = model.module.genotype()
        logging.info('genotype = %s', genotype)

    ops = []
    for cell_type in ['normal', 'reduce']:
        for edge in range(model.module.num_edges):
            ops.append(['{}_{}_{}'.format(cell_type, edge, i) for i in
                        range(0, model.module.num_ops)])
    ops = np.concatenate(ops)
    lr = args.learning_rate
    accum_shaps = [1e-3 * torch.randn(model.module.num_edges, model.module.num_ops).cuda(),
                   1e-3 * torch.randn(model.module.num_edges, model.module.num_ops).cuda()]

    for epoch in range(start_epoch, args.epochs):
        scheduler.step()
        current_lr = scheduler.get_lr()[0]
        logging.info('Epoch: %d lr: %e', epoch, current_lr)
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
            print(optimizer)
        genotype = model.module.genotype()
        logging.info('genotype = %s', genotype)

        if epoch >= args.epochs // 2:

            shap_normal, shap_reduce = shap_estimation(valid_queue, model, criterion,
                                                       ops, num_samples=args.samples, threshold=args.threshold)

            accum_shaps = change_alpha(model, [shap_normal, shap_reduce], accum_shaps, momentum=args.shapley_momentum,
                                       step_size=args.step_size)

        train_acc, train_obj = train(train_queue, model, optimizer, criterion)
        logging.info('Train_acc %f', train_acc)

        # validation
        if epoch >= 47:
            valid_acc, valid_obj = infer(infer_queue, model, criterion)
            logging.info('Valid_acc %f', valid_acc)

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'alpha': model.module.arch_parameters()
        }, False, args.save)

    genotype = model.module.genotype()
    logging.info('genotype = %s', genotype)


def remove_players(normal_weights, reduce_weights, op):
    selected_cell = str(op.split('_')[0])
    selected_eid = int(op.split('_')[1])
    opid = int(op.split('_')[-1])
    proj_mask = torch.ones_like(normal_weights[selected_eid])
    proj_mask[opid] = 0
    if selected_cell in ['normal']:
        if not ((normal_weights[selected_eid][:opid] == 0.0).all() and (
                normal_weights[selected_eid][opid + 1:] == 0.0).all()):
            normal_weights[selected_eid] = normal_weights[selected_eid] * proj_mask
    else:
        if not ((reduce_weights[selected_eid][:opid] == 0.0).all() and (
                reduce_weights[selected_eid][opid + 1:] == 0.0).all()):
            reduce_weights[selected_eid] = reduce_weights[selected_eid] * proj_mask


def shap_estimation(valid_queue, model, criterion, players, num_samples, threshold=0.5):
    """
    Implementation of Monte-Carlo sampling of Shapley value for operation importance evaluation

    """

    permutations = None
    n = len(players)
    sv_acc = np.zeros((n, num_samples))

    with torch.no_grad():

        if permutations is None:
            # Keep the same permutations for all batches
            permutations = [np.random.permutation(n) for _ in range(num_samples)]

        for j in range(num_samples):
            x, y = next(iter(valid_queue))
            x, y = x.cuda(), y.cuda(non_blocking=True)
            logits = model.module(x, weights_dict=None)
            ori_prec1, = utils.accuracy(logits, y, topk=(1,))

            normal_weights = model.module.get_projected_weights('normal')
            reduce_weights = model.module.get_projected_weights('reduce')

            acc = ori_prec1.data.item()
            print('MC sampling %d times' % (j+1))

            for idx, i in enumerate(permutations[j]):

                remove_players(normal_weights, reduce_weights, players[i])
                logits = model.module(x, weights_dict={'normal': normal_weights, 'reduce': reduce_weights})
                prec1, = utils.accuracy(logits, y, topk=(1,))
                new_acc = prec1.item()
                delta_acc = acc - new_acc
                sv_acc[i][j] = delta_acc
                acc = new_acc
                # print(players[i], delta_acc)

                if acc < threshold * ori_prec1:
                    break

        result = np.mean(sv_acc, axis=-1) - np.std(sv_acc, axis=-1)
        shap_acc = np.reshape(result, (2, model.module.num_edges, model.module.num_ops))
        shap_normal, shap_reduce = shap_acc[0], shap_acc[1]
        return shap_normal, shap_reduce


def change_alpha(model, shap_values, accu_shap_values, momentum=0.8, step_size=0.1):
    assert len(shap_values) == len(model.module.arch_parameters())

    shap = [torch.from_numpy(shap_values[i]).cuda() for i in range(len(model.module.arch_parameters()))]

    for i, params in enumerate(shap):
        mean = params.data.mean()
        std = params.data.std()
        params.data.add_(-mean).div_(std)

    updated_shap = [
        accu_shap_values[i] * momentum \
        + shap[i] * (1. - momentum)
        for i in range(len(model.module.arch_parameters()))]

    for i, p in enumerate(model.module.arch_parameters()):
        p.data.add_((step_size * updated_shap[i]).to(p.device))

    return updated_shap


def train(train_queue, model, optimizer, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()

