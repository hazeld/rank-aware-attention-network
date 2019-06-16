import argparse
import torch
import time
import shutil

from dataset import SkillDataSet
from model import RAAN
from opts import parser
from losses import RankingAttentionLoss 

from tensorboardX import SummaryWriter

best_prec = 0

def main():
    global args, best_prec, writer
    args = parser.parse_args()

    writer = SummaryWriter('_'.join((args.run_folder, 'attention', str(args.attention), 'filters',
                                     str(args.num_filters), 'diversity', str(args.diversity_loss),
                                     str(args.lambda_param), 'disparity', str(args.disparity_loss),
                                     'rank_aware', str(args.rank_aware_loss), 'lr', str(args.lr))))

    if args.rank_aware_loss:
        num_attention_branches = 2
        models = {'pos': None, 'neg': None}
    else:
        num_attention_branches = 1
        models = {'att': None}
    for k in models.keys():
        models[k] = RAAN(args.num_samples, args.attention, args.num_filters).cuda()
    if args.disparity_loss or args.rank_aware_loss:
        model_uniform = RAAN(args.num_samples, False, 1).cuda()

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

    criterion = torch.nn.MarginRankingLoss(margin=args.m1).cuda()

    if args.disparity_loss or args.rank_aware_loss:
        attention_params = []
        model_params = []
        for model in models.values():
            for name, param in model.named_parameters():
                if param.requires_grad and 'att' in name:
                    attention_params.append(param)
                else:
                    model_params.append(param)
        optimizer = torch.optim.Adam(list(model_uniform.parameters()) + model_params, args.lr)
        optimizer_attention = torch.optim.Adam(attention_params, args.lr*0.1)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)

    if args.evaluate:
        validate(val_loader, models, criterion, 0)

    phase = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.disparity_loss:
            phase = train_with_uniform(train_loader, models, model_uniform, criterion,
                                       optimizer, optimizer_attention,
                                       epoch, phase=phase)
        else:
            train(train_loader, models, criterion, optimizer, epoch)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec = validate(val_loader, models, criterion, (epoch + 1))
            is_best = prec > best_prec
            best_prec = max(prec, best_prec)
            checkpoint_dict = {'epoch': epoch + 1, 'best_prec': best_prec}
            for k in models.keys():
                checkpoint_dict['state_dict_' + k] = models[k].state_dict(),
            if args.disparity_loss or args.rank_aware_loss:
                checkpoint_dict['state_dict_uniform'] = model_uniform.state_dict(),
            save_checkpoint(checkpoint_dict, is_best)
    writer.close()

def train(train_loader, models, criterion, optimizer, epoch, shuffle=True, phase=0):
    av_meters = {'batch_time': AverageMeter(), 'data_time': AverageMeter(), 'losses': AverageMeter(),
                 'ranking_losses': AverageMeter(), 'diversity_losses': AverageMeter(),
                 'acc': AverageMeter()}
    model = models[models.keys()[0]]
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
            all_losses += args.lambda_param*(div_loss_att1 + div_loss_att2)
            
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


def train_with_uniform(train_loader, models, model_uniform, criterion, optimizer, optimizer_attention, epoch, shuffle=True, phase=0):
    av_meters = {'batch_time': AverageMeter(), 'data_time': AverageMeter(), 'losses': AverageMeter(),
                 'ranking_losses': AverageMeter(), 'ranking_losses_uniform': AverageMeter(),
                 'diversity_losses': AverageMeter(), 'disparity_losses': AverageMeter(),
                 'rank_aware_losses': AverageMeter(), 'acc': AverageMeter(), 'acc_uniform': AverageMeter()}
    
    for k in models.keys():
        models[k].train()
    model_uniform.train()
    
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

        all_output1, all_output2, output1, output2, att1, att2 = {}, {}, {}, {}, {}, {}
        for k in models.keys():
            all_output1[k], att1[k] = models[k](input_var1)
            all_output2[k], att2[k] = models[k](input_var2)
            output1[k] = all_output1[k].mean(dim=1)
            output2[k] = all_output2[k].mean(dim=1)
        output1_uniform, _ = model_uniform(input_var1)
        output2_uniform, _ = model_uniform(input_var2)
        output1_uniform = output1_uniform.mean(dim=1)
        output2_uniform = output2_uniform.mean(dim=1)

        ranking_loss = 0
        disparity_loss = 0
        for k in models.keys():
            ranking_loss += criterion(output1[k], output2[k], target)
            disparity_loss += multi_rank_loss(all_output1[k], all_output2[k], output1_uniform,
                                                output2_uniform, target, args.m2)
        ranking_loss_uniform = criterion(output1_uniform, output2_uniform, target)
        if args.rank_aware_loss:
            rank_aware_loss = multi_rank_loss(all_output1['pos'], all_output2['neg'], output1_uniform,
                                              output2_uniform, target, args.m3)

        if args.diversity_loss:
            div_loss_att1, div_loss_att2 = 0, 0
            for k in models.keys():
                div_loss_att1 += diversity_loss(att1[k])
                div_loss_att2 += diversity_loss(att2[k])

        all_losses = 0
        if phase == 0:
            all_losses += ranking_loss
            all_losses += ranking_loss_uniform
        else:
            all_losses += disparity_loss
            if args.rank_aware_loss:
                all_losses += rank_aware_loss
            if args.diversity_loss:
                all_losses += args.lambda_param*(div_loss_att1 + div_loss_att2)
        # measure accuracy and backprop
        output1_all = torch.zeros(output1[list(models.keys())[0]].data.shape).cuda()
        output2_all = torch.zeros(output2[list(models.keys())[0]].data.shape).cuda()
        for k in models.keys():
            output1_all += output1[k].data
            output2_all += output2[k].data
        prec = accuracy(output1_all, output2_all)
        prec_uniform = accuracy(output1_uniform.data, output2_uniform.data)

        all_losses.backward()
        
        if phase == 0:
            optimizer.step()
            optimizer.zero_grad()
            phase = 1
        else:
            optimizer_attention.step()
            optimizer_attention.zero_grad()
            phase = 0

        # record losses
        av_meters['ranking_losses'].update(ranking_loss.item(), input1.size(0)*len(models.keys()))
        av_meters['ranking_losses_uniform'].update(ranking_loss_uniform.item(), input1.size(0))
        av_meters['disparity_losses'].update(disparity_loss.item(), input1.size(0*len(models.keys())))
        if args.diversity_loss:
            av_meters['diversity_losses'].update(div_loss_att1.item() + div_loss_att2.item(),
                                                input1.size(0)*2*len(models.keys()))
        if args.rank_aware_loss:
            av_meters['rank_aware_losses'].update(rank_aware_loss.item(), input1.size(0))
        av_meters['losses'].update(all_losses.data.item(), input1.size(0))
        av_meters['acc'].update(prec, input1.size(0))
        av_meters['acc_uniform'].update(prec_uniform, input1.size(0))

        # measure elapsed time
        av_meters['batch_time'].update(time.time() - end)
        end = time.time()

        if i % (args.print_freq) == 0:
            console_log_train(av_meters, epoch, i, len(train_loader), )

    tensorboard_log_with_uniform(av_meters, 'train', epoch)
    return phase

    
def validate(val_loader, models, criterion, epoch):
    av_meters = {'batch_time': AverageMeter(), 'losses': AverageMeter(),
                 'ranking_losses': AverageMeter(), 'diversity_losses': AverageMeter(),
                 'acc': AverageMeter()}

    # switch to evaluate mode
    for k in models.keys():
        models[k].eval()

    end = time.time()
    for i, (input1, input2) in enumerate(val_loader):
        input_var1 = torch.autograd.Variable(input1.cuda())
        input_var2 = torch.autograd.Variable(input2.cuda())

        all_output1, all_output2, output1, output2, att1, att2 = {}, {}, {}, {}, {}, {}
        for k in models.keys():
            all_output1[k], att1[k] = models[k](input_var1)
            all_output2[k], att2[k] = models[k](input_var2)
            output1[k] = all_output1[k].mean(dim=1)
            output2[k] = all_output2[k].mean(dim=1)

        labels = torch.ones(input1.size(0)).cuda()
        target = torch.autograd.Variable(labels)

        ranking_loss = 0
        for k in models.keys():
            ranking_loss += criterion(output1[k], output2[k], target)
        all_losses = ranking_loss
        if args.diversity_loss:
            div_loss_att1, div_loss_att2 = 0, 0
            for k in models.keys():
                div_loss_att1 += diversity_loss(att1[k])
                div_loss_att2 += diversity_loss(att2[k])
            all_losses += args.lambda_param*(div_loss_att1 + div_loss_att2)
        
        # measure accuracy
        # measure accuracy and backprop
        output1_all = torch.zeros(output1[list(models.keys())[0]].data.shape).cuda()
        output2_all = torch.zeros(output2[list(models.keys())[0]].data.shape).cuda()
        for k in models.keys():
            output1_all += output1[k].data
            output2_all += output2[k].data
        prec = accuracy(output1_all, output2_all)

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
    filename = '_'.join((args.snapshot_pref, 'attention', str(args.attention), 'filters',
                         str(args.num_filters), 'diversity', str(args.diversity_loss), 'disparity',
                         str(args.disparity_loss), 'rank_aware', str(args.rank_aware_loss),
                         str(args.lambda_param), 'lr', str(args.lr), filename))
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

def multi_rank_loss(input_a_1, input_a_2, input_b_1, input_b_2, target, margin):
    inter1, _ = torch.min((input_a_1 - input_a_2), dim=1)
    inter2 = (input_b_1 - input_b_2)
    inter = -target * (inter1.view(-1) - inter2.view(-1)) + torch.ones(input_a_1.size(0)).cuda()*margin
    losses = torch.max(torch.zeros(input_a_1.size(0)).cuda(), inter)
    return losses.sum()/input_a_1.size(0)

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

def tensorboard_log_with_uniform(av_meters, mode, epoch):
    tensorboard_log(av_meters, mode, epoch)
    writer.add_scalar(mode+'/disparity_loss', av_meters['disparity_losses'].avg, epoch)
    writer.add_scalar(mode+'/ranking_loss_uniform', av_meters['ranking_losses_uniform'].avg, epoch)
    writer.add_scalar(mode+'/acc_uniform', av_meters['acc_uniform'].avg, epoch)
    writer.add_scalar(mode+'/rank_aware_loss', av_meters['rank_aware_losses'].avg, epoch)


def data_augmentation(input_var1, input_var2):
    noise = torch.autograd.Variable(torch.normal(torch.zeros(input_var1.size()[1],
                                                             input_var1.size()[2]),
                                                 0.01)).cuda()
    input_var1 = torch.add(input_var1, noise)
    input_var2 = torch.add(input_var2, noise)
    return input_var1, input_var2
    
if __name__ == '__main__':
    main()


    
