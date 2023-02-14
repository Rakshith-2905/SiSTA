import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist

from sklearn.metrics import confusion_matrix

#from memo.cifar.utils.third_party import aug
from utils_memo import aug 

from celebahq_dataloader import RefImgs,CelebaHQDataset,train_transform,test_transform,JOJOGanDataset,JOJOGanDatasetMultipleImages
from image_NRC_target import Entropy, op_copy,lr_scheduler,print_args,test_target
import wandb
from randconv import get_random_module
def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


def aug_randconv(img,conv_module): 
    
    img =conv_module(img)
    return img

def adapt_single(netF,netB,netC, image, optimizer, criterion,
                 niter, batch_size, prior_strength,randconv_module=None):
    netF.eval()
    netB.eval()
    netC.eval()
    if prior_strength < 0:
        nn.BatchNorm2d.prior = 1
    else:
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)
    if len(image.shape) ==3: 
        image= image.unsqueeze(0).cuda()
    randconv_module = randconv_module.cuda()
    for iteration in range(niter):
        
        if randconv_module is None:
            inputs = [aug(image) for _ in range(batch_size)]
        else: 
            randconv_module.randomize()
            inputs = [aug_randconv(image,randconv_module) for _ in range(batch_size)]
        inputs = torch.stack(inputs).squeeze().cuda()
        optimizer.zero_grad()
        outputs = netC(netB(netF(inputs)))

        loss, logits = criterion(outputs)
        loss.backward()
        optimizer.step()
    nn.BatchNorm2d.prior = 1


def test_single(netF,netB,netC, image, label,prior_strength):
    netF.eval()
    netB.eval()
    netC.eval()

    if prior_strength < 0:
        nn.BatchNorm2d.prior = 1
    else:
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)
    transform = image_test() 
    inputs = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = netC(netB(netF(inputs.cuda())))
        _, predicted = outputs.max(1)
        confidence = nn.functional.softmax(outputs, dim=1).squeeze()[predicted].item()
    correctness = 1 if predicted.item() == label else 0
    nn.BatchNorm2d.prior = 1
    return correctness, confidence

def cal_acc(loader, netF, netB, netC,prior_strength, flag=False):
    start_test = True
    netF.eval()
    netB.eval()
    netC.eval()
    if prior_strength < 0:
        nn.BatchNorm2d.prior = 1
    else:
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength) + 1
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
   
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def data_load(args):
    dsets = {}
    dset_loaders={}
    train_bs = args.batch_size
    test_domain=args.names[args.t]
    if args.augmix:
        dsets['target']= RefImgs(domain=test_domain,
            
            transform= None,
            return_index=True)
    else:
        dsets['target']= RefImgs(domain=test_domain,
            
            transform= train_transform,
            return_index=True)

    dsets['test'] = CelebaHQDataset(attribute=args.attribute,\
        data_dir= args.data_dir,
        domain= test_domain,
        train=False,
        transform= test_transform,
        return_index=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=False, num_workers=args.worker, drop_last=False)
    return dsets,dset_loaders



def train_target(args):
    
    dsets,dset_loaders = data_load(args)
    ## set base network
    randconv_module = get_random_module(mixing=True)
    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

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



    teset= dsets['target']
    correct=[]
    for i in range(len(teset)):
        modelpath = args.output_dir_src + '/source_F.pt'
        netF.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir_src + '/source_B.pt'
        netB.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir_src + '/source_C.pt'
        netC.load_state_dict(torch.load(modelpath))
        netC.eval()

        inputs_test, label,idx = teset[i-1]
        if args.augmix:
            adapt_single(netF,netB,netC, inputs_test, optimizer, marginal_entropy,
                    1, 64, 16)
        else: 
            adapt_single(netF,netB,netC, inputs_test, optimizer, marginal_entropy,
                    2, 64, 16,randconv_module=randconv_module)

      
        #correct.append(test_single(netF,netB, netC,inputs_test, label, 16)[0])
    acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB,
                                            netC,16,flag= True)
    
    log_str = 'Task: {}, Accuracy on target = {:.2f}%'.format(
                args.name,  acc_s_te
            ) + '\n' + 'T: ' + acc_list
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')
  

    if args.issave:   
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        
    return netF, netB, netC

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
    parser.add_argument('--seed', type=int, default=2021, help="random seed")

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
    parser.add_argument('--issave', type=bool, default=False)
    parser.add_argument('--attribute',type=str,default='Smiling')
    parser.add_argument('--data_dir',type=str,default='/home/kowshik/generative_sfda/DATA/CelebA-HQ-split')
    parser.add_argument('--variant', type=str, default= 'prune_rewind',\
                    choices = ['interp_concat','prune_rewind','interp_mixup','direct_target'])
    parser.add_argument('--prune',type=bool,default=True)
    parser.add_argument('--rewind',type=bool,default=False)
    parser.add_argument('--output', type=str, default='CHECKPOINTS/checkpoints_randconv')
    parser.add_argument('--augmix',type=bool,default=False)
   # parser.add_argument('--output', type=str, default='CHECKPOINTS/prune_rewind/checkpoints_JOJOGAN_NRC_prune_rewind')
    

    args = parser.parse_args()
    

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
        for attribute in ['Smiling','Male','High_Cheekbones', 'Mouth_Slightly_Open','Arched_Eyebrows', 'Big_Lips']:
            args.attribute = attribute
            args.output_dir_src = osp.join(args.output_src, args.dset,args.attribute)
            args.name_src = args.attribute
            args.output_dir = osp.join(
                args.output,  args.dset, args.attribute,
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
            #run= wandb.init(project=args.variant+'_'+args.attribute+'_'+names[args.t],reinit=True,mode='disabled')
            train_target(args)
            #run.finish()
        #test_target(args)
