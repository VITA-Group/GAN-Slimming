import argparse, itertools, os, time
import numpy as np 
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage import img_as_ubyte

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F 
from PIL import Image
import torch
import torch.nn as nn

from models.models import Generator, Discriminator
from utils.utils import *
from utils.perceptual import *
from utils.fid_score import calculate_fid_given_paths
from datasets.datasets import ImageDataset, PairedImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--cpus', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--gpu', default='7')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--wd', type=float, default=1e-3, help='weight decay')
parser.add_argument('--momentum', default=0.5, type=float, help='momentum')
parser.add_argument('--dataset', type=str, default='horse2zebra', choices=['summer2winter_yosemite', 'horse2zebra'])
parser.add_argument('--task', type=str, default='A2B', choices=['A2B', 'B2A'])
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--beta', type=float, default=0.001, help='minimax loss weight')
parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
parser.add_argument('--lc', default='vgg', choices=['vgg', 'mse'], help='G content loss. vgg: perceptual; mse: mse')
parser.add_argument('--base_model_str', help='from which subnet to finetune')
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

quant = True if 'GS8' in args.base_model_str else False
subnet_model_path = os.path.join('subnet_structures', args.dataset, args.task, args.base_model_str, 'pth')

## Networks
# G:
dim_lst_path = os.path.join(subnet_model_path, 'epoch%d_netG.npy' % 199)
netG = Generator(args.input_nc, args.output_nc, dim_lst=np.load(dim_lst_path), quant=quant).cuda()
# D:
netD = Discriminator(args.input_nc).cuda()

## results_dir:
optimizer_str = 'adam_lr%s_wd%s' % (args.lr, args.wd)
loss_str = 'beta%s_%s' % (args.beta, args.lc)
results_dir = os.path.join('finetune_results', args.dataset, args.task, args.base_model_str, 
    '%s_%s' % (optimizer_str, loss_str))
img_dir = os.path.join(results_dir, 'img')
pth_dir = os.path.join(results_dir, 'pth')
create_dir(img_dir), create_dir(pth_dir)

# Optimizers 
optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.5, 0.999)) # lr=1e-3
optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.5, 0.999)) # lr=1e-3

# LR schedulers:
lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, args.epochs)
lr_scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, args.epochs)

# load trained models:
if args.resume:
    last_epoch, loss_G_lst, loss_G_perceptual_lst, loss_G_GAN_lst, loss_D_lst, best_FID = load_ckpt_finetune(
        netG, netD, 
        optimizer_G, optimizer_D, 
        lr_scheduler_G, lr_scheduler_D, 
        path=os.path.join(results_dir, 'pth', 'latest.pth')
    )
    start_epoch = last_epoch + 1
else:
    # load sub G extracted from latest.pth
    g_path = os.path.join(subnet_model_path, 'epoch%d_netG.pth' % 199) 
    netG.load_state_dict(torch.load(g_path))
    print('load G from %s' % g_path)
    # load full D directly from latest.pth
    d_path = os.path.join('results', args.dataset, args.task, args.base_model_str, 'pth', 'latest.pth')
    netD.load_state_dict(torch.load(d_path)['netD'])
    print('load D from %s' % d_path)
    start_epoch = 0
    best_FID = 1e9
    loss_G_lst, loss_G_perceptual_lst, loss_G_GAN_lst, loss_D_lst = [], [], [], []

# Dataset loader: img shape=(256,256)
dataset_dir = os.path.join(foreign_dir, 'datasets', args.dataset)
soft_data_dir = os.path.join(foreign_dir, 'train_set_result', args.dataset)
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ] # (0,1) -> (-1,1)
dataloader = DataLoader(
    PairedImageDataset(dataset_dir, soft_data_dir, transforms_=transforms_, mode=args.task), 
    batch_size=args.batch_size, shuffle=True, num_workers=args.cpus, drop_last=True)
dataloader_test = DataLoader(
    ImageDataset(os.path.join(dataset_dir, 'test', source_str), transforms_=transforms_), 
    batch_size=1, shuffle=False, num_workers=args.cpus)

# FID img dirs:
test_img_generation_dir_temp = os.path.join(results_dir, 'test_set_generation_temp')
create_dir(test_img_generation_dir_temp)
img_paths_for_FID = [test_img_generation_dir_temp, os.path.join(dataset_dir, 'train', target_str)]

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
input_source = Tensor(args.batch_size, args.input_nc, args.size, args.size)
input_target = Tensor(args.batch_size, args.output_nc, args.size, args.size)
target_real = Variable(Tensor(args.batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(args.batch_size).fill_(0.0), requires_grad=False)
fake_img_buffer = ReplayBuffer()
input_source_test = Tensor(1, args.input_nc, args.size, args.size)


# preload models:
if args.lc == 'vgg':
    vgg = VGGFeature().cuda()

###### Training ######
N = len(dataloader)
print('N:', N) # 1334
for epoch in range(start_epoch, args.epochs):
    start_time = time.time()
    netG.train(), netG.train()
    # reset loss to 0
    loss_G_meter, loss_G_perceptual_meter, loss_G_GAN_meter, loss_D_meter = \
        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    for i, batch in enumerate(dataloader):
        # Set model input
        input_img = Variable(input_source.copy_(batch[source_str]))
        teacher_output_img = Variable(input_target.copy_(batch[target_str]))

        ###### G ######
        optimizer_G.zero_grad()

        student_output_img = netG(input_img)

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
        loss_G = loss_G_perceptual + args.beta * loss_G_GAN
        loss_G.backward()
        
        optimizer_G.step()
        
        # append loss:
        loss_G_meter.append(loss_G.item())
        loss_G_perceptual_meter.append(loss_G_perceptual.item())
        loss_G_GAN_meter.append(loss_G_GAN.item())
        
        if i % 50 == 0:
            if args.lc == 'vgg':
                out_str_G = 'epoch %d-%d-G: perceptual %.4f (content %.4f, style %.4f)' % (
                    epoch, i, loss_G_perceptual.data, loss_G_content.data, loss_G_style.data * 1e5)
            elif args.lc == 'mse':
                out_str_G = 'epoch %d-%d-G: mse %.4f' % (
                    epoch, i, loss_G_perceptual.data)
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
        loss_D = args.beta * (loss_D_real + loss_D_fake)*0.5
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

    ## save best FID
    if epoch > int(args.epochs/2) and epoch % 10 == 0 or epoch == args.epochs - 1:    
        # generate images:   
        for i, batch_test in enumerate(dataloader_test):
            # Set model input
            input_img_test = Variable(input_source_test.copy_(batch_test))
            # Generate output
            output_img_test = 0.5*(netG(input_img_test).data + 1.0)
            # Save image files
            save_image( output_img_test, os.path.join(test_img_generation_dir_temp, '%04d.png' % (i+1)) )

            sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader_test)))
        print()
        # find FID:
        FID = calculate_fid_given_paths(img_paths_for_FID)
        if FID < best_FID:
            best_FID = FID
            torch.save(netG.state_dict(), os.path.join(pth_dir, 'best_FID_netG.pth'))
            # torch.save(netD.state_dict(), os.path.join(pth_dir, 'best_FID_netD.pth'))
            best_FID_epoch = epoch

        f_FID = open(os.path.join(results_dir, 'FID_log.txt'), 'a+')
        f_FID.write('epoch %d: FID %s\n' % (epoch, FID))
        f_FID.close()
    ## End save best FID

    # save latest:
    save_ckpt_finetune(epoch, netG, netD, 
        optimizer_G, optimizer_D, 
        lr_scheduler_G, lr_scheduler_D,
        loss_G_lst, loss_G_perceptual_lst, loss_G_GAN_lst, loss_D_lst, 
        best_FID, 
        path=os.path.join(pth_dir, 'latest.pth'))
###### End Training ######
