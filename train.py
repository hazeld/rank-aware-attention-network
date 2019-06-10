import argparse
import torch
import time
import shutil

from dataset import SkillDataSet
from model import RAAN
from opts import parser

from tensorboardX import SummaryWriter

best_prec = 0

def main():
    global args, best_prec, writer
    args = parser.parse_args()

    writer = SummaryWriter('_'.join((args.run_folder, 'attention', str(args.attention), 'lr', str(args.lr))))

    model = RAAN(args.num_samples, args.attention, args.num_filters)
    model = model.cuda()

    train_loader = torch.utils.data.DataLoader(
        SkillDataSet(args.root_path, args.train_list, ftr_tmpl='{}_{}.npz'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        SkillDataSet(args.root_path, args.val_list, ftr_tmpl='{}_{}.npz'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    criterion = torch.nn.MarginRankingLoss(margin=1.0).cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec = validate(val_loader, model, criterion, (epoch + 1))
            is_best = prec > best_prec
            best_prec = max(prec, best_prec)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec': best_prec,
            }, is_best)
    writer.close()

def train(train_loader, model, criterion, optimizer, epoch, shuffle=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()

    end = time.time()
    optimizer.zero_grad()
    for i, (input1, input2) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input_var1 = torch.autograd.Variable(input1.cuda(), requires_grad=True)
        input_var2 = torch.autograd.Variable(input2.cuda(), requires_grad=True)
        ## add small amount of gaussian noise to features for data augmentation
        if args.transform:
            noise = torch.autograd.Variable(torch.normal(torch.zeros(input_var1.size()[1],
                                                                     input_var1.size()[2]),
                                                         0.01)).cuda()
            input_var1 = torch.add(input_var1, noise)
            input_var2 = torch.add(input_var2, noise)
            
        labels = torch.ones(input1.size(0)).cuda()

        output1, att1 = model(input_var1)
        output2, att2 = model(input_var2)
        
        target  = torch.autograd.Variable(labels, requires_grad=False)
        loss = criterion(output1, output2, target)

        # measure accuracy and record loss
        prec = accuracy(output1.data, output2.data)
        all_losses = loss
        losses.update(all_losses.data.item(), input1.size(0))
        acc.update(prec, input1.size(0))

        all_losses.backward()

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % (args.print_freq) == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec: {acc.val:.3f} ({acc.avg:.3f})'.format(
                       epoch, i, len(train_loader), lr=optimizer.param_groups[-1]['lr'],
                       batch_time=batch_time, data_time=data_time, loss=losses, acc=acc)))
        writer.add_scalar('train/total_loss', losses.avg, epoch)
        writer.add_scalar('train/acc', acc.avg, epoch)

def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input1, input2) in enumerate(val_loader):
        input_var1 = torch.autograd.Variable(input1.cuda())
        input_var2 = torch.autograd.Variable(input2.cuda())

        output1, att1 = model(input_var1)
        output2, att2 = model(input_var2)

        labels = torch.ones(input1.size(0)).cuda()
        target = torch.autograd.Variable(labels)
        loss = criterion(output1, output2, target)

        # measure accuracy and record loss
        prec = accuracy(output1.data, output2.data)
        all_losses = loss
        losses.update(all_losses.data[0], input1.size(0))
        acc.update(prec, input1.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % (args.print_freq) == 0:
            print(('Test: [{0}/{1}\t'
                   'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec {acc.val} ({acc.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc)))

    print(('Testing Results: Acc {acc.avg:.3f} Loss {loss.avg:.5f}'
           .format(acc=acc, loss=losses)))
    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/acc', acc.avg, epoch)
    return acc.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, 'attention', str(args.attention), 'lr', str(args.lr), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)

class AverageMeter(object):
    """Compute and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output1, output2):
    """Computes the % of correctly ordered pairs"""
    pred1 = output1
    pred2 = output2
    correct = torch.gt(pred1, pred2)
    return float(correct.sum())/correct.size(0)

if __name__ == '__main__':
    main()


    
