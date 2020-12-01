import argparse, itertools, os, time
import numpy as np 
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage import img_as_ubyte

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F 
from PIL import Image
import torch
import torch.nn as nn

from models.models import Generator, Discriminator
from utils.utils import *
from utils.perceptual import *
from datasets.datasets import ImageDataset, PairedImageDataset


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='7')
parser.add_argument('--cpus', default=4)
parser.add_argument('--batch_size', '-b', default=8, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--lrw', type=float, default=1e-5, help='learning rate for G')
parser.add_argument('--lrgamma', type=float, default=1e-1, help='learning rate for gamma (pruning)')
parser.add_argument('--wd', type=float, default=1e-3, help='weight decay for G')
parser.add_argument('--momentum', default=0.5, type=float, help='momentum')
parser.add_argument('--dataset', type=str, default='horse2zebra', choices=['summer2winter_yosemite', 'horse2zebra'])
parser.add_argument('--task', type=str, default='A2B', choices=['A2B', 'B2A'])
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--rho', type=float, default=0, help='l1 loss weight')
parser.add_argument('--beta', type=float, default=20, help='GAN loss weight')
parser.add_argument('--lc', default='vgg', choices=['vgg', 'mse'], help='G content loss. vgg: perceptual; mse: mse')
parser.add_argument('--quant', action='store_true', help='enable quantization (for both activation and weight)')
parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
args = parser.parse_args()
if args.task == 'A2B':
    source_str, target_str = 'A', 'B'
else:
    source_str, target_str = 'B', 'A'
foreign_dir = './'
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

## results_dir:
method_str = ('GS8' if args.quant else 'GS32')
gamma_optimizer_str = 'sgd_mom%s_lrgamma%s' % (args.momentum, args.lrgamma)
W_optimizer_str = 'adam_lrw%s_wd%s' % (args.lrw, args.wd)
opt_str = 'e%d-b%d' % (args.epochs, args.batch_size)
loss_str = 'rho%s_beta%s_%s' % (args.rho, args.beta, args.lc)
results_dir = os.path.join('results', args.dataset, args.task, '%s_%s_%s_%s_%s' % (
    method_str, loss_str, opt_str, gamma_optimizer_str, W_optimizer_str))
img_dir = os.path.join(results_dir, 'img')
pth_dir = os.path.join(results_dir, 'pth')
create_dir(img_dir), create_dir(pth_dir)

## Networks
# G:
netG = Generator(args.input_nc, args.output_nc, quant=args.quant).cuda()
# D:
netD = Discriminator(args.input_nc).cuda()

# param list:
parameters_G, parameters_D, parameters_gamma = [], [], []
for name, para in netG.named_parameters():
    if 'weight' in name and para.ndimension() == 1:
        parameters_gamma.append(para)
    else:
        parameters_G.append(para)
for name, para in netD.named_parameters():
    # print(name, para.size(), para.ndimension()) 
    parameters_D.append(para)
print('parameters_gamma:', len(parameters_gamma))

# Optimizers:
optimizer_gamma = torch.optim.SGD(parameters_gamma, lr=args.lrgamma, momentum=args.momentum)
optimizer_G = torch.optim.Adam(parameters_G, lr=args.lrw, weight_decay=args.wd, betas=(0.5, 0.999)) # lr=1e-3
optimizer_D = torch.optim.Adam(parameters_D, lr=args.lrw, weight_decay=args.wd, betas=(0.5, 0.999)) # lr=1e-3

# LR schedulers:
lr_scheduler_gamma = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gamma, args.epochs)
lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, args.epochs)
lr_scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, args.epochs)

# load pretrained:
if args.resume:
    last_epoch, loss_G_lst, loss_G_perceptual_lst, loss_G_GAN_lst, loss_D_lst, channel_number_lst = load_ckpt(
        netG, netD, 
        optimizer_G, optimizer_D, optimizer_gamma, 
        lr_scheduler_G, lr_scheduler_D, lr_scheduler_gamma, 
        path=os.path.join(results_dir, 'pth', 'latest.pth')
    )
    start_epoch = last_epoch + 1
else:
    dense_model_folder = 'pretrained_dense_model_quant' if args.quant else 'pretrained_dense_model'
    g_path = os.path.join(foreign_dir, dense_model_folder, args.dataset, 'pth', 'netG_%s_epoch_%d.pth' % (args.task, 199) )
    netG.load_state_dict(torch.load(g_path))
    print('load G from %s' % g_path)
    d_path = os.path.join(foreign_dir, dense_model_folder, args.dataset, 'pth', 'netD_%s_epoch_%d.pth' % (target_str, 199) )
    netD.load_state_dict(torch.load(d_path))
    print('load D from %s' % d_path)
    start_epoch = 0
    loss_G_lst, loss_G_perceptual_lst, loss_G_GAN_lst, loss_D_lst, channel_number_lst = [], [], [], [], []

# Dataset loader: image shape=(256,256)
dataset_dir = os.path.join(foreign_dir, 'datasets', args.dataset)
soft_data_dir = os.path.join(foreign_dir, 'train_set_result', args.dataset) 
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ] # (0,1) -> (-1,1)
dataloader = DataLoader(
    PairedImageDataset(dataset_dir, soft_data_dir, transforms_=transforms_, mode=args.task), 
    batch_size=args.batch_size, shuffle=True, num_workers=args.cpus, drop_last=True)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
input_source = Tensor(args.batch_size, args.input_nc, args.size, args.size)
input_target = Tensor(args.batch_size, args.output_nc, args.size, args.size)
target_real = Variable(Tensor(args.batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(args.batch_size).fill_(0.0), requires_grad=False)
fake_img_buffer = ReplayBuffer()

# perceptual loss models:
vgg = VGGFeature().cuda()

###### Training ######
print('dataloader:', len(dataloader)) # 1334
for epoch in range(start_epoch, args.epochs):
    start_time = time.time()
    netG.train(), netD.train()
    # define average meters:
    loss_G_meter, loss_G_perceptual_meter, loss_G_GAN_meter, loss_D_meter = \
        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    for i, batch in enumerate(dataloader):
        # Set model input
        input_img = Variable(input_source.copy_(batch[source_str])) # X 
        teacher_output_img = Variable(input_target.copy_(batch[target_str])) # Gt(X)

        # print('input_img:', input_img.size(), torch.max(input_img), torch.min(input_img))
        # print('teacher_output_img:', teacher_output_img.size(), torch.max(teacher_output_img), torch.min(teacher_output_img))

        ###### G ######
        optimizer_G.zero_grad()
        optimizer_gamma.zero_grad()

        student_output_img = netG(input_img) # Gs(X)

        # perceptual loss
        if args.lc == 'vgg': 
            student_output_vgg_features = vgg(student_output_img)
            teacher_output_vgg_features = vgg(teacher_output_img)

            loss_G_perceptual, loss_G_content, loss_G_style = \
                perceptual_loss(student_output_vgg_features, teacher_output_vgg_features)
        elif args.lc == 'mse':
            loss_G_perceptual = F.mse_loss(student_output_img, teacher_output_img)

        # GAN loss (G part):
        pred_student_output_img = netD(student_output_img)
        loss_G_GAN = torch.nn.MSELoss()(pred_student_output_img, target_real)

        # Total G loss
        loss_G = args.beta * loss_G_perceptual + loss_G_GAN
        loss_G.backward()
        
        optimizer_G.step()
        optimizer_gamma.step()

        # append loss:
        loss_G_meter.append(loss_G.item())
        loss_G_perceptual_meter.append(loss_G_perceptual.item())
        loss_G_GAN_meter.append(loss_G_GAN.item())

        # proximal gradient for channel pruning:
        current_lr = lr_scheduler_gamma.get_lr()[0]
        for name, m in netG.named_modules():
            if isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
                m.weight.data = soft_threshold(m.weight.data, th=float(args.rho) * float(current_lr))
        
        if i % 50 == 0:
            if args.lc == 'vgg':
                out_str_G = 'epoch %d-%d-G: perceptual %.4f (content %.4f, style %.4f) | gamma lr %.4f' % (
                    epoch, i, loss_G_perceptual.data, loss_G_content.data, loss_G_style.data * 1e5, current_lr)
            elif args.lc == 'mse':
                out_str_G = 'epoch %d-%d-G: mse %.4f | gamma lr %.4f' % (
                    epoch, i, loss_G_perceptual.data, current_lr)
            print(out_str_G)
        ###### End G ######


        ###### D ######
        optimizer_D.zero_grad()
        
        # real loss:
        pred_teacher_output_img = netD(teacher_output_img)
        loss_D_real = torch.nn.MSELoss()(pred_teacher_output_img, target_real)

        # Fake loss
        student_output_img_buffer_pop = fake_img_buffer.push_and_pop(student_output_img)
        pred_student_output_img = netD(student_output_img_buffer_pop.detach())
        loss_D_fake = torch.nn.MSELoss()(pred_student_output_img, target_fake)

        # Total loss
        loss_D = (loss_D_real + loss_D_fake)*0.5
        loss_D.backward()

        optimizer_D.step()

        # append loss:
        loss_D_meter.append(loss_D.item())
        ###### End D ######


    ## at the end of each epoch
    netG.eval(), netG.eval()
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()
    lr_scheduler_gamma.step()

    print('time: %.2f' % (time.time()-start_time))
    print(args)

    # plot training loss:
    losses = {}
    losses['loss_G'] = (loss_G_lst, loss_G_meter.avg)
    losses['loss_G_perceptual'] = (loss_G_perceptual_lst, loss_G_perceptual_meter.avg)
    losses['loss_G_GAN'] = (loss_G_GAN_lst, loss_G_GAN_meter.avg)
    losses['loss_D'] = (loss_D_lst, loss_D_meter.avg)
    for key in losses:
        losses[key][0].append(losses[key][1])
        plt.plot(losses[key][0])
        plt.savefig(os.path.join(results_dir, '%s.png' % key))
        plt.close()

    if epoch % 10 == 0 or epoch == args.epochs - 1:
        # save imgs:
        images={'input_img': input_img, 'teacher_output_img': teacher_output_img, 'student_output_img': student_output_img}
        for key in images:
            img_np = images[key].detach().cpu().numpy()
            img_np = np.moveaxis(img_np, 1, -1)
            img_np = (img_np + 1) / 2 # (-1,1) -> (0,1)
            img_big = fourD2threeD(img_np, n_row=4)
            print(key, img_big.shape, np.amax(img_big), np.amin(img_big))
            imsave(os.path.join(img_dir, 'epoch%d_%s.png' % (epoch, key)), img_as_ubyte(img_big))

    if epoch % 20 == 0 or epoch == args.epochs - 1:    
        # Save models checkpoints
        torch.save(netG.state_dict(), os.path.join(pth_dir, 'epoch%d_netG.pth' % epoch))

        # TBD: save best FID

    # channel number:
    channel_number_lst.append(none_zero_channel_num(netG))
    plt.plot(channel_number_lst)
    plt.savefig(os.path.join(results_dir, 'channel_number.png'))
    plt.close()

    # save latest:
    save_ckpt(epoch, netG, netD, 
        optimizer_G, optimizer_D, optimizer_gamma, 
        lr_scheduler_G, lr_scheduler_D, lr_scheduler_gamma,
        loss_G_lst, loss_G_perceptual_lst, loss_G_GAN_lst, loss_D_lst, channel_number_lst, 
        path=os.path.join(pth_dir, 'latest.pth'))
###### End Training ######
