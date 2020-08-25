import argparse, itertools, os
import numpy as np 
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage import img_as_ubyte

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn

from models.models import Generator, Discriminator, Conv2dQuant, ConvTrans2dQuant
from utils.utils import *


input_nc, output_nc = 3, 3
# model_str = 'adam_lr1e-05_wd0.001_sgd_mom0.5_lr0.1_de100_rho0.0008_beta100000.0_vgg_vgg'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='horse2zebra', choices=['summer2winter_yosemite', 'horse2zebra'])
parser.add_argument('--task', type=str, default='A2B', choices=['A2B', 'B2A'])
parser.add_argument('--epoch', type=int, default=199)
parser.add_argument('--gpu', default='7')
parser.add_argument('--model_str')
args = parser.parse_args()
print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

quant = True if 'GS8' in args.model_str else False

# load dense model:
g_path = os.path.join('cp_results', args.dataset, args.task, args.model_str, 'pth', 
    'epoch%d_netG.pth' % args.epoch)
dense_model = Generator(input_nc, output_nc, quant=quant)
dense_model.load_state_dict(torch.load(g_path))
dense_model = nn.DataParallel(dense_model)
# measure_model(dense_model, 256, 256) # 54168.000000M

# get channel numbers:
dim_lst = []
for m in dense_model.modules():
    if isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
        gamma = m.weight.data.cpu().numpy()
        channel_num = np.sum(gamma!=0)
        dim_lst.append(channel_num)
print(dim_lst)

# construct subnet:

sub_model = Generator(input_nc, output_nc, dim_lst=dim_lst, quant=quant)
measure_model(sub_model, 256, 256)
print('sub_model:', type(sub_model))

print("#parameters: ", model_param_num(sub_model))

# save model structure npy:
save_dir = os.path.join('subnet_structures_cp', args.dataset, args.task, args.model_str, 'pth')
create_dir(save_dir)
np.save(os.path.join(save_dir, 'epoch%d_netG.npy' % args.epoch), np.array(dim_lst))


## load parameters from dense model to submodel:
# sub_model.load_state_dict(dense_model.state_dict(), strict=False)

weight_list = {}
pre_m = None
pre_name = ""
input_nc = [0, 1, 2]
for layer, (name, m) in enumerate(dense_model.named_modules()):
    if isinstance(m, nn.InstanceNorm2d) and m.weight is not None and pre_m is not None:
        gamma = m.weight.data
        channel_indicator = torch.nonzero(gamma).flatten().cpu().numpy().tolist()
        # print("oc indicator", channel_indicator)
        # print("original ms", pre_m.weight.data.shape)
        # print("original bias", pre_m.bias.data.shape)

        if isinstance(pre_m, nn.Conv2d) or isinstance(pre_m, Conv2dQuant):
            temp = pre_m.weight.data[channel_indicator,:,:,:]
            weight_list[pre_name[7:]] = (temp[:, input_nc, :, :], pre_m.bias.data[channel_indicator])
        elif isinstance(pre_m, ConvTrans2dQuant) or isinstance(pre_m, nn.ConvTranspose2d):
            temp = pre_m.weight.data[:,channel_indicator,:, :]
            weight_list[pre_name[7:]] = (temp[input_nc, :, :, :], pre_m.bias.data[channel_indicator])
        weight_list[name[7:]] = (m.weight.data[channel_indicator], m.bias.data[channel_indicator])
        pre_m = m
        pre_name = name
        input_nc = list(channel_indicator)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) \
        or isinstance(m, Conv2dQuant) or isinstance(m, ConvTrans2dQuant):
        pre_m = m
        pre_name = name
    elif isinstance(m, nn.InstanceNorm2d) and m.weight is None:
        weight_list[pre_name[7:]] = (pre_m.weight.data[:, input_nc, :, :], pre_m.bias.data)
        input_nc = [i for i in range(pre_m.weight.data.shape[0])]
        pre_m = m
        pre_name = name
    elif isinstance(m, nn.Tanh) and isinstance(pre_m, nn.Conv2d) and name.strip().split(".")[-1] == "27":
        weight_list[pre_name[7:]] = (pre_m.weight.data[:, input_nc, :, :], pre_m.bias.data)
        pre_m = m
        pre_name = name
        
for name, m in sub_model.named_modules():
    if name in weight_list:
        m.weight.data.copy_(weight_list[name][0])
        m.bias.data.copy_(weight_list[name][1])
        
# sub_model = nn.DataParallel(sub_model)

torch.save(sub_model.state_dict(), os.path.join(save_dir, 'epoch%d_netG.pth' % args.epoch))
