'''
Elements for perceptual loss
'''

from utils.vgg import Vgg16
import torch.nn as nn
import torch

class VGGFeature(nn.Module):
    def __init__(self):
        super(VGGFeature, self).__init__()
        self.add_module('vgg', Vgg16())
    def __call__(self,x):
        x = (x.clone()+1.)/2. # [-1,1] -> [0,1]
        x_vgg = self.vgg(x)
        return x_vgg

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram
    

def perceptual_loss(vgg_features1, vgg_features2, beta=1e5, layer='relu1_2'):
    '''
    Calculate perceptual loss based on two vgg features.

    Agrs:
        vgg_features1, vgg_features2: vgg features. 

    Output:
        loss_perceptual, loss_content, loss_style: scalar Tensor. 
    '''
    if layer=='relu1_2':
        loss_content = torch.nn.L1Loss()(vgg_features1.relu1_2, vgg_features2.relu1_2)
    elif layer=='relu2_2':
        loss_content = torch.nn.L1Loss()(vgg_features1.relu2_2, vgg_features2.relu2_2)
    elif layer=='relu3_3':
        loss_content = torch.nn.L1Loss()(vgg_features1.relu3_3, vgg_features2.relu3_3)
    loss_style = 0
    for _, (vf_g, vf_c) in enumerate(zip(vgg_features1, vgg_features2)):
        # print('vf_g:', vf_g.size())
        gm_g, gm_c = gram_matrix(vf_g), gram_matrix(vf_c)
        # print('gm_g:', gm_g.size())
        loss_style += nn.functional.mse_loss(gm_g, gm_c)
    loss_perceptual = loss_content + 1e5 * loss_style
    
    return loss_perceptual, loss_content, loss_style
