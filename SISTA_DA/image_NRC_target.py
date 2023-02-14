import argparse
import copy
import math
import os
import os.path as osp
import pdb
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
import loss
import network
#from celeba_dataloader import celeba_train,celeba_val,celeba_test,attribute_dict
from celebahq_dataloader import (CelebaHQDataset, JOJOGanDataset,
                                 JOJOGanDatasetMultipleImages,
                                 JOJOGANPruneRewind, test_transform,
                                 train_transform)
from data_list import ImageList, ImageList_idx
from image_source import cal_acc


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter)**(-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer




def data_load(args):
    dsets = {}
    dset_loaders={}
    train_bs = args.batch_size
    test_domain=args.names[args.t]
    if args.variant == 'prune_rewind':
        dsets['target'] = JOJOGANPruneRewind(domain=test_domain,
            interp_weights= args.interp_weights,
            transform= train_transform,
            return_index=True,
            prune = args.prune,
            rewind= args.rewind)#
    elif args.variant =='interp_concat':
        dsets['target'] = JOJOGanDataset(domain=test_domain,
            interp_weights= args.interp_weights,
            transform= train_transform,
            return_index=True,
            train_size=args.train_size)#
    elif args.variant =='direct_target':
    
        dsets['target'] = CelebaHQDataset(attribute=args.attribute,
            data_dir= args.data_dir,
            domain=test_domain,
            train=False,
            transform= train_transform,
            return_index=True) # Durect test set
    
    dsets['test'] = CelebaHQDataset(attribute=args.attribute,\
        data_dir= args.data_dir,
        domain= test_domain,
        train=False,
        transform= test_transform,
        return_index=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=False, num_workers=args.worker, drop_last=False)
    return dset_loaders



def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier,
                                   feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    param_group = []
    param_group_c=[]
    for k, v in netF.named_parameters():
        #if k.find('bn')!=-1:
        if True:
            param_group += [{'params': v, 'lr': args.lr *0.1}]

    for k, v in netB.named_parameters():
        if True:
            param_group += [{'params': v, 'lr': args.lr * 0.1}]
    for k, v in netC.named_parameters():
        param_group_c += [{'params': v, 'lr': args.lr * 0.01}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)


    #building feature bank and score bank
    loader = dset_loaders["target"]
    num_sample=len(loader.dataset)
    fea_bank=torch.randn(num_sample,256)
    score_bank = torch.randn(num_sample, args.class_num).cuda()

    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            indx=data[-1]
            #labels = data[1]
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            output_norm=F.normalize(output)
            outputs = netC(output)
            outputs=nn.Softmax(dim=1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  #.cpu()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()
    acc_log=0
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        #output_re = softmax_out.unsqueeze(1)

        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()

            distance = output_f_@fea_bank.T
            _, idx_near = torch.topk(distance,
                                    dim=-1,
                                    largest=True,
                                    k=args.K+1)
            idx_near = idx_near[:, 1:]  #batch x K
            score_near = score_bank[idx_near]    #batch x K x C

            fea_near = fea_bank[idx_near]  #batch x K x num_dim
            fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0],-1,-1) # batch x n x dim
            distance_ = torch.bmm(fea_near, fea_bank_re.permute(0,2,1))  # batch x K x n
            _,idx_near_near=torch.topk(distance_,dim=-1,largest=True,k=args.KK+1)  # M near neighbors for each of above K ones
            idx_near_near = idx_near_near[:,:,1:] # batch x K x M
            tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
            match = (
                idx_near_near == tar_idx_).sum(-1).float()  # batch x K
            weight = torch.where(
                match > 0., match,
                torch.ones_like(match).fill_(0.1))  # batch x K

            weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                    args.KK)  # batch x K x M
            weight_kk = weight_kk.fill_(0.1)

            # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
            #weight_kk[idx_near_near == tar_idx_]=0

            score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
            #print(weight_kk.shape)
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                    -1)  # batch x KM

            score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
                                               args.class_num)  # batch x KM x C

            score_self = score_bank[tar_idx]


        # nn of nn
        output_re = softmax_out.unsqueeze(1).expand(-1, args.K * args.KK,
                                                    -1)  # batch x C x 1
        const = torch.mean(
            (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
             weight_kk.cuda()).sum(1)) # kl_div here equals to dot product since we do not use log for score_near_kk
        extended_nn_loss = torch.mean(const)


        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, args.K,
                                                         -1)  # batch x K x C

        nn_loss= torch.mean((
            F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) *
            weight.cuda()).sum(1))
        wandb.log({'nn_loss':nn_loss })

        # self, if not explicitly removing the self feature in expanded neighbor then no need for this
        #loss += -torch.mean((softmax_out * score_self).sum(-1))


        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(msoftmax *
                                    torch.log(msoftmax + args.epsilon))
        loss =extended_nn_loss +nn_loss+gentropy_loss

        optimizer.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_c.step()
        wandb.log({'extended_nn_loss':extended_nn_loss,'nn_loss':nn_loss,'gentropy Loss':gentropy_loss,'total_loss':loss })


        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            
            acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB,
                                            netC,args,flag= True)
            # # log_str = 'Task: {}, Iter:{}/{}; Accuracy on target = {:.2f}%'.format(
            # #     args.name, iter_num, max_iter, acc_s_te
            # # ) + '\n' + 'T: ' + acc_list
            # log_str = 'Task: {}, Iter:{}/{}; Accuracy on target = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            
            log_str = 'Task: {}, Iter:{}/{}; Accuracy on target = {:.2f}%'.format(
                args.name, iter_num, max_iter, acc_s_te
            ) + '\n' + 'T: ' + acc_list
            #log_str = 'Task: {}, Iter:{}/{}; Accuracy on target = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            wandb.log({'Target Accuracy': acc_s_te, 'True_negative':float( acc_list.split(' ')[0]), 'True_positive':float( acc_list.split(' ')[1])})
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()
            netC.train()

            if acc_s_te>acc_log:
                acc_log=acc_s_te
                torch.save(
                    netF.state_dict(),
                    osp.join(args.output_dir, "target_F_" + '2021_'+str(args.K) + ".pt"))
                torch.save(
                    netB.state_dict(),
                    osp.join(args.output_dir,
                                "target_B_" + '2021_' + str(args.K) + ".pt"))
                torch.save(
                    netC.state_dict(),
                    osp.join(args.output_dir,
                                "target_C_" + '2021_' + str(args.K) + ".pt"))


    return netF, netB, netC

def test_target(args):

    dset_loaders = data_load(args)
    netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    
    args.modelpath = osp.join(args.output_dir, "target_F_" + '2021_'+str(args.K) + ".pt")
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = osp.join(args.output_dir,
                                "target_B_" + '2021_' + str(args.K) + ".pt")  
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = osp.join(args.output_dir,
                                "target_C_" + '2021_' + str(args.K) + ".pt")
    netC.load_state_dict(torch.load(args.modelpath))

    netF.eval()
    netB.eval()
    netC.eval()

    
    acc_s_te, acc_list  = cal_acc(dset_loaders['test'], netF, netB, netC, args,True)
    
    log_str = 'Task: {}; Accuracy on target = {:.2f}%'.format(
        args.name,  acc_s_te
    ) + '\n' + 'T: ' + acc_list
    
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)




def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neighbors')
    parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='8',
                        help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=3, help="target")
    parser.add_argument('--max_epoch',
                        type=int,
                        default=15,
                        help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="batch_size")
    parser.add_argument('--worker',
                        type=int,
                        default=4,
                        help="number of workers")
    parser.add_argument('--dset', type=str, default='celebahq', choices=['VISDA-C','celeba','celebahq', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--net',
                        type=str,
                        default='resnet50')
    parser.add_argument('--seed', type=int, default=2022, help="random seed")

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--KK', type=int, default=5)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer',
                        type=str,
                        default="wn",
                        choices=["linear", "wn"])
    parser.add_argument('--classifier',
                        type=str,
                        default="bn",
                        choices=["ori", "bn"])
    
    parser.add_argument('--output_src', type=str, default='CHECKPOINTS/checkpoints_2class_source/')
    parser.add_argument('--tag', type=str, default='selfplus')
    parser.add_argument('--da',
                        type=str,
                        default='uda')
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--attribute',type=str,default='Smiling')
    parser.add_argument('--data_dir',type=str,default='/home/kowshik/generative_sfda/DATA/CelebA-HQ-split')
    parser.add_argument('--variant', type=str, default= 'interp_concat',\
                    choices = ['interp_concat','prune_rewind','interp_mixup','direct_target'])
    parser.add_argument('--prune',type=bool,default=False)
    parser.add_argument('--rewind',type=bool,default=True)
    parser.add_argument('--train_size',type=int,default=500)
    #parser.add_argument('--output', type=str, default='CHECKPOINTS/prune_rewind/prune_rewind/checkpoints_JOJOGAN_NRC_prune_rewind')
   # parser.add_argument('--output', type=str, default='CHECKPOINTS/prune_rewind/checkpoints_JOJOGAN_NRC_prune_rewind')
    

    args = parser.parse_args()
    if args.variant == 'prune_rewind':
        args.interp_weights = [0,1]
        
        args.output = 'CHECKPOINTS/prune_rewind/prune_followed_rewind/checkpoints_JOJOGAN_NRC_prune_followed_rewind'
    elif 'interp' in args.variant:
        args.interp_weights = [0] # or [0,][0,2,4]
        args.output = 'CHECKPOINTS/checkpoints_JOJOGAN_NRC'#_interp_weights_02

    elif args.variant == 'direct_target':
        args.output = 'CHECKPOINTS/checkpoints_target_NRC'
    else:
        pass
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'visda-2017':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'celebahq':
        names = ['photo', 'watercolor', 'color_sketch','pencil_sketch']
        args.names = names
        args.class_num = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    for t in [1,2,3]:
        args.t = t
        for attribute in ['Arched_Eyebrows']:#, 'Big_Lips']:#,'High_Cheekbones', 'Mouth_Slightly_Open','Arched_Eyebrows', 'Big_Lips']:
            args.attribute = attribute
            args.output_dir_src = osp.join(args.output_src, args.dset,args.attribute)
            args.name_src = args.attribute
            args.output_dir = osp.join(
                args.output,  args.dset, 'train_size_'+str(args.train_size),args.attribute,
                names[args.s] + names[args.t])
            args.name = names[args.s][0].upper() + names[args.t][0].upper()

            if not osp.exists(args.output_dir):
                os.system('mkdir -p ' + args.output_dir)
            if not osp.exists(args.output_dir):
                os.mkdir(args.output_dir)

            args.out_file = open(
                osp.join(args.output_dir, 'log_target.txt'), 'w')
            args.out_file.write(print_args(args) + '\n')
            args.out_file.flush()
            run= wandb.init(project=args.variant+'_'+args.attribute+'_'+names[args.t],reinit=True)
            train_target(args)
            run.finish()
        #test_target(args)
