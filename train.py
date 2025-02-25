import time
import os
import torch.autograd
from skimage import io
from torch import optim, nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F

working_path = os.path.abspath('.')

from utils.loss import weighted_bce, feature_Similarity
from utils.utils import binary_accuracy as accuracy
from utils.utils import AverageMeter

###################### Data and Model ########################
from models.TFNet import TFNet as Net
NET_NAME = 'TFNet'

from datasets import dataset_read as RS
DATA_NAME = 'cropland'
#from datasets import WHU_CD_list as RS
#DATA_NAME = 'WHU_CD_0.05'
###################### Data and Model ########################

########################## Parameters ########################
args = {
    'train_batch_size': 32,
    'val_batch_size': 32,
    'lr': 0.1,
    'epochs': 80,
    'gpu': True,
    'dev_id': 0,
    'multi_gpu': None,  #"0,1,2,3",
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'print_freq': 50,
    'predict_step': 5,
    'crop_size': 256,
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME),
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'xxx.pth')}
########################## Parameters ########################

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
writer = SummaryWriter(args['log_dir'])


def main():
    net = Net()
    #net.load_state_dict(torch.load(args['load_path']), strict=False)
    # if args['multi_gpu']:
    #     net = torch.nn.DataParallel(net, [int(id) for id in args['multi_gpu'].split(',')])
    net.to(device=torch.device('cuda', int(args['dev_id'])))

    train_set = RS.RS('train', random_crop=True, crop_nums=10, crop_size=args['crop_size'], random_flip=True) #
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=0, shuffle=True)
    val_set = RS.RS('test', sliding_crop=False, crop_size=args['crop_size'], random_flip=False)
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=0, shuffle=False)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), args['lr'],
                          weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)

    # optimizer = optim.AdamW(net.parameters(), args['lr'])

    train(train_loader, net, optimizer, val_loader)
    writer.close()
    print('Training finished.')

def train(train_loader, net, optimizer, val_loader):
    bestF = 0.0
    bestacc = 0.0
    bestIoU = 0.0
    bestloss = 1.0
    bestaccT = 0.0

    curr_epoch = 0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    criterion_sem = feature_Similarity().to(torch.device('cuda', int(args['dev_id'])))
    # criterion_sem = feature_Similarity().to(torch.device('cuda', int(args['dev_id'])))
    while True:
        torch.cuda.empty_cache()
        net.train()
        start = time.time()
        acc_meter = AverageMeter()
        train_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            adjust_lr(optimizer, running_iter, all_iters, args)
            imgs_A, mask, boundary = data           #
            if args['gpu']:
                imgs_A = imgs_A.to(torch.device('cuda', int(args['dev_id']))).float()
                mask = mask.to(torch.device('cuda', int(args['dev_id']))).float().unsqueeze(1)
                boundary = boundary.to(torch.device('cuda', int(args['dev_id']))).float().unsqueeze(1)

            optimizer.zero_grad()
            ##原始
            outA, outB = net(imgs_A)
            assert outA.shape[1] == 1
            loss_A = F.binary_cross_entropy_with_logits(outA, mask)
            cetrion = weighted_bce()
            loss_B = cetrion(outB, boundary)
            loss_t = criterion_sem(outA, outB, mask)
            loss = loss_A + loss_B + loss_t

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            optimizer.step()

            label_mask = mask.cpu().detach().numpy()
            outputs = outA.cpu().detach()
            preds = F.sigmoid(outputs).numpy()
            acc_curr_meter = AverageMeter()
            for (pred, label) in zip(preds, label_mask):
                acc, precision, recall, F1, IoU = accuracy(pred, label)
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_loss.update(loss.cpu().detach().numpy())
            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_loss.val, acc_meter.val * 100))
                writer.add_scalar('train loss', train_loss.val, running_iter)
                loss_rec = train_loss.val
                writer.add_scalar('train accuracy', acc_meter.val, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        val_F, val_acc, val_IoU, val_loss = validate(val_loader, net, curr_epoch)
        if val_F > bestF:
            bestF = val_F
            bestacc = val_acc
            bestIoU = val_IoU
            torch.save(net.state_dict(), os.path.join(args['chkpt_dir'], NET_NAME + '_e%d_OA%.2f_F%.2f_IoU%.2f.pth' % (
            curr_epoch, val_acc * 100, val_F * 100, val_IoU * 100)))
        if acc_meter.avg > bestaccT: bestaccT = acc_meter.avg
        print('[epoch %d/%d %.1fs] Best rec: Train %.2f, Val %.2f, F1 score: %.2f IoU %.2f' \
              % (curr_epoch, args['epochs'], time.time() - begin_time, bestaccT * 100, bestacc * 100, bestF * 100,
                 bestIoU * 100))
        curr_epoch += 1
        if curr_epoch >= args['epochs']:
            return


def validate(val_loader, net, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    Acc_meter = AverageMeter()

    for vi, data in enumerate(val_loader):
        imgs_A, mask, boundary = data

        if args['gpu']:
            imgs_A = imgs_A.to(torch.device('cuda', int(args['dev_id']))).float()
            mask = mask.to(torch.device('cuda', int(args['dev_id']))).float().unsqueeze(1)
            boundary = boundary.to(torch.device('cuda', int(args['dev_id']))).float().unsqueeze(1)

        with torch.no_grad():
            outA, outB = net(imgs_A)
            loss1 = F.binary_cross_entropy_with_logits(outA, mask)
            cetrion1 = weighted_bce()
            loss2 = cetrion1(outB, boundary)
            loss = loss1+loss2
        val_loss.update(loss.cpu().detach().numpy())

        outputs = outA.cpu().detach()
        label_mask = mask.cpu().detach().numpy()
        preds = F.sigmoid(outputs).numpy()
        for (pred, label) in zip(preds, label_mask):
            acc, precision, recall, F1, IoU = accuracy(pred, label)
            F1_meter.update(F1)
            Acc_meter.update(acc)
            IoU_meter.update(IoU)

        if curr_epoch % args['predict_step'] == 0 and vi == 0:
            pred_color = RS.Index2Color(preds[0].squeeze())
            io.imsave(os.path.join(args['pred_dir'], NET_NAME + '.png'), pred_color)
            print('Prediction saved!')

    curr_time = time.time() - start
    print('%.1fs Val loss %.2f Acc %.2f F %.2f' % (
    curr_time, val_loss.average(), Acc_meter.average() * 100, F1_meter.average() * 100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Accuracy', Acc_meter.average(), curr_epoch)

    return F1_meter.avg, Acc_meter.avg, IoU_meter.avg, val_loss.avg


def adjust_lr(optimizer, curr_iter, all_iter, args):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** 3.0)
    running_lr = args['lr'] * scale_running_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    main()
