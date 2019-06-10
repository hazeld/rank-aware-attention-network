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

    writer = SummaryWriter('_'.join((args.run_folder, 'attention', str(args.attention), 'filters', str(args.num_filters), 'diversity', str(args.diversity_loss), str(args.beta), 'lr', str(args.lr))))

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
    av_meters = {'batch_time': AverageMeter(), 'data_time': AverageMeter(), 'losses': AverageMeter(),
                 'ranking_losses': AverageMeter(), 'diversity_losses': AverageMeter(),
                 'acc': AverageMeter()}
    
    model.train()

    end = time.time()
    optimizer.zero_grad()
    for i, (input1, input2) in enumerate(train_loader):
        # measure data loading time
        av_meters['data_time'].update(time.time() - end)
        input_var1 = torch.autograd.Variable(input1.cuda(), requires_grad=True)
        input_var2 = torch.autograd.Variable(input2.cuda(), requires_grad=True)
        ## add small amount of gaussian noise to features for data augmentation
        if args.transform:
            input_var1, input_var2 = data_augmentation(input_var1, input_var2)
            
        labels = torch.ones(input1.size(0)).cuda()
        target  = torch.autograd.Variable(labels, requires_grad=False)

        output1, att1 = model(input_var1)
        output2, att2 = model(input_var2)
        
        ranking_loss = criterion(output1, output2, target)
        all_losses = ranking_loss
        if args.diversity_loss:
            div_loss_att1 = diversity_loss(att1)
            div_loss_att2 = diversity_loss(att2)
            all_losses += args.beta*(div_loss_att1 + div_loss_att2)
            
        # measure accuracy and backprop
        prec = accuracy(output1.data, output2.data)

        all_losses.backward()

        optimizer.step()
        optimizer.zero_grad()

        # record losses
        av_meters['ranking_losses'].update(ranking_loss.item(), input1.size(0))
        if args.diversity_loss:
            av_meters['diversity_losses'].update(div_loss_att1.item() + div_loss_att2.item(),
                                                input1.size(0)*2)
        av_meters['losses'].update(all_losses.data.item(), input1.size(0))
        av_meters['acc'].update(prec, input1.size(0))

        # measure elapsed time
        av_meters['batch_time'].update(time.time() - end)
        end = time.time()

        if i % (args.print_freq) == 0:
            console_log_train(av_meters, epoch, i, len(train_loader), )

    tensorboard_log(av_meters, 'train', epoch) 

def validate(val_loader, model, criterion, epoch):
    av_meters = {'batch_time': AverageMeter(), 'losses': AverageMeter(),
                 'ranking_losses': AverageMeter(), 'diversity_losses': AverageMeter(),
                 'acc': AverageMeter()}

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

        ranking_loss = criterion(output1, output2, target)
        all_losses = ranking_loss
        if args.diversity_loss:
            div_loss_att1 = diversity_loss(att1)
            div_loss_att2 = diversity_loss(att2)
            all_losses += args.beta*(div_loss_att1 + div_loss_att2)
        
        # measure accuracy
        prec = accuracy(output1.data, output2.data)

        # record losses
        av_meters['ranking_losses'].update(ranking_loss.item(), input1.size(0))
        if args.diversity_loss:
            av_meters['diversity_losses'].update(div_loss_att1.item() + div_loss_att2.item(), input1.size(0)*2)
        av_meters['losses'].update(all_losses.data.item(), input1.size(0))
        av_meters['acc'].update(prec, input1.size(0))

        # measure elapsed time
        av_meters['batch_time'].update(time.time() - end)
        end = time.time()

        if i % (args.print_freq) == 0:
            console_log_test(av_meters, i, len(val_loader))

    print(('Testing Results: Acc {acc.avg:.3f} Loss {loss.avg:.5f}'
           .format(acc=av_meters['acc'], loss=av_meters['losses'])))
    tensorboard_log(av_meters, 'val', epoch)
    
    return av_meters['acc'].avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, 'attention', str(args.attention), 'filters', str(args.num_filters), 'diversity', str(args.diversity_loss), str(args.beta), 'lr', str(args.lr), filename))
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

def diversity_loss(attention):
    attention_t = torch.transpose(attention, 1, 2)
    num_features = attention.shape[1]
    res = torch.matmul(attention_t.view(-1, args.num_filters, num_features), attention.view(-1, num_features, args.num_filters)) - torch.eye(args.num_filters).cuda()
    res = res.view(-1, args.num_filters*args.num_filters)
    return torch.norm(res, p=2, dim=1).sum() / attention.size(0)

def console_log_train(av_meters, epoch, iter, epoch_len):
    print(('Epoch: [{0}][{1}/{2}]\t'
           'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
           'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
           'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
           'Prec: {acc.val:.3f} ({acc.avg:.3f})'.format(
               epoch, iter, epoch_len, batch_time=av_meters['batch_time'],
               data_time=av_meters['data_time'], loss=av_meters['losses'],
               acc=av_meters['acc'])))

def console_log_test(av_meters, iter, test_len):
    print(('Test: [{0}/{1}\t'
           'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
           'Prec {acc.val} ({acc.avg:.3f})'.format(
               iter, test_len, batch_time=av_meters['batch_time'], loss=av_meters['losses'],
               acc=av_meters['acc'])))

def tensorboard_log(av_meters, mode, epoch):
    writer.add_scalar(mode+'/total_loss', av_meters['losses'].avg, epoch)
    writer.add_scalar(mode+'/ranking_loss', av_meters['ranking_losses'].avg, epoch)
    writer.add_scalar(mode+'/diversity_loss', av_meters['diversity_losses'].avg, epoch)
    writer.add_scalar(mode+'/acc', av_meters['acc'].avg, epoch)

def data_augmentation(input_var1, input_var2):
    noise = torch.autograd.Variable(torch.normal(torch.zeros(input_var1.size()[1],
                                                             input_var1.size()[2]),
                                                 0.01)).cuda()
    input_var1 = torch.add(input_var1, noise)
    input_var2 = torch.add(input_var2, noise)
    return input_var1, input_var2
    
if __name__ == '__main__':
    main()


    