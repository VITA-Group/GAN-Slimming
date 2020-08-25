import random, time, os, datetime, sys

from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn.functional import normalize
import numpy as np

import matplotlib.pyplot as plt


class AverageMeter:
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.values = []
		self.counter = 0

	def append(self, val):
		self.values.append(val)
		self.counter += 1

	@property
	def val(self):
		return self.values[-1]

	@property
	def avg(self):
		return sum(self.values) / len(self.values)

	@property
	def last_avg(self):
		if self.counter == 0:
			return self.latest_avg
		else:
			self.latest_avg = sum(self.values[-self.counter:]) / self.counter
			self.counter = 0
			return self.latest_avg


def save_ckpt(epoch, netG, netD, 
	optimizer_G, optimizer_D, optimizer_gamma, 
	lr_scheduler_G, lr_scheduler_D, lr_scheduler_gamma,
	loss_G_lst, loss_G_perceptual_lst, loss_G_GAN_lst, loss_D_lst, channel_number_lst, 
	path):

	ckpt = {
		'epoch': epoch,
		'netG': netG.state_dict(),
		'netD': netD.state_dict(),
		'optimizer_G': optimizer_G.state_dict(),
		'optimizer_D': optimizer_D.state_dict(),
		'optimizer_gamma': optimizer_gamma.state_dict(),
		'lr_scheduler_G': lr_scheduler_G.state_dict(),
		'lr_scheduler_D': lr_scheduler_D.state_dict(),
		'lr_scheduler_gamma': lr_scheduler_gamma.state_dict(),
		'loss_G_lst': loss_G_lst,
		'loss_G_perceptual_lst': loss_G_perceptual_lst,
		'loss_G_GAN_lst': loss_G_GAN_lst,
		'loss_D_lst': loss_D_lst,
		'channel_number_lst': channel_number_lst,
	}
	torch.save(ckpt, path)

def load_ckpt(netG, netD, 
	optimizer_G, optimizer_D, optimizer_gamma, 
	lr_scheduler_G, lr_scheduler_D, lr_scheduler_gamma, path):
	
	if not os.path.isfile(path):
		raise Exception('No such file: %s' % path)
	print("===>>> loading checkpoint from %s" % path)
	ckpt = torch.load(path)
	
	epoch = ckpt['epoch']
	loss_G_lst = ckpt['loss_G_lst']
	loss_G_perceptual_lst = ckpt['loss_G_perceptual_lst']
	loss_G_GAN_lst = ckpt['loss_G_GAN_lst']
	loss_D_lst = ckpt['loss_D_lst']
	channel_number_lst = ckpt['channel_number_lst']
	# best_FID = ckpt['best_FID']
	
	netG.load_state_dict(ckpt['netG'])
	netD.load_state_dict(ckpt['netD'])
	optimizer_G.load_state_dict(ckpt['optimizer_G'])
	optimizer_D.load_state_dict(ckpt['optimizer_D'])
	optimizer_gamma.load_state_dict(ckpt['optimizer_gamma'])
	lr_scheduler_G.load_state_dict(ckpt['lr_scheduler_G'])
	lr_scheduler_D.load_state_dict(ckpt['lr_scheduler_D'])
	lr_scheduler_gamma.load_state_dict(ckpt['lr_scheduler_gamma'])
	
	return epoch, loss_G_lst, loss_G_perceptual_lst, loss_G_GAN_lst, loss_D_lst, channel_number_lst

def save_ckpt_finetune(epoch, netG, netD, 
	optimizer_G, optimizer_D, 
	lr_scheduler_G, lr_scheduler_D,
	loss_G_lst, loss_G_perceptual_lst, loss_G_GAN_lst, loss_D_lst, 
	best_FID, 
	path):

	ckpt = {
		'epoch': epoch,
		'netG': netG.state_dict(),
		'netD': netD.state_dict(),
		'optimizer_G': optimizer_G.state_dict(),
		'optimizer_D': optimizer_D.state_dict(),
		'lr_scheduler_G': lr_scheduler_G.state_dict(),
		'lr_scheduler_D': lr_scheduler_D.state_dict(),
		'loss_G_lst': loss_G_lst,
		'loss_G_perceptual_lst': loss_G_perceptual_lst,
		'loss_G_GAN_lst': loss_G_GAN_lst,
		'loss_D_lst': loss_D_lst,
		'best_FID': best_FID,
	}
	torch.save(ckpt, path)

def load_ckpt_finetune(netG, netD, 
	optimizer_G, optimizer_D, 
	lr_scheduler_G, lr_scheduler_D, 
	path):

	if not os.path.isfile(path):
		raise Exception('No such file: %s' % path)
	print("===>>> loading checkpoint from %s" % path)
	ckpt = torch.load(path)
	epoch = ckpt['epoch']
	loss_G_lst = ckpt['loss_G_lst']
	loss_G_perceptual_lst = ckpt['loss_G_perceptual_lst']
	loss_G_GAN_lst = ckpt['loss_G_GAN_lst']
	loss_D_lst = ckpt['loss_D_lst']

	best_FID = ckpt['best_FID']

	netG.load_state_dict(ckpt['netG'])
	netD.load_state_dict(ckpt['netD'])

	optimizer_G.load_state_dict(ckpt['optimizer_G'])
	optimizer_D.load_state_dict(ckpt['optimizer_D'])

	lr_scheduler_G.load_state_dict(ckpt['lr_scheduler_G'])
	lr_scheduler_D.load_state_dict(ckpt['lr_scheduler_D'])
	return epoch, loss_G_lst, loss_G_perceptual_lst, loss_G_GAN_lst, loss_D_lst, best_FID


def tensor2image(tensor):
	image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
	if image.shape[0] == 1:
		image = np.tile(image, (3,1,1))
	return image.astype(np.uint8)      
	   

class ReplayBuffer():
	'''
	follow Shrivastava et al.’s strategy: 
	update D using a history of generated images, rather than the ones produced by the latest generators. 
	'''
	def __init__(self, max_size=50):
		assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
		self.max_size = max_size
		self.data = []

	def push_and_pop(self, data):
		to_return = []
		for element in data.data:
			element = torch.unsqueeze(element, 0)
			if len(self.data) < self.max_size:
				self.data.append(element)
				to_return.append(element)
			else:
				if random.uniform(0,1) > 0.5:
					i = random.randint(0, self.max_size-1)
					to_return.append(self.data[i].clone())
					self.data[i] = element
				else:
					to_return.append(element)
		return Variable(torch.cat(to_return))

class LambdaLR():
	def __init__(self, n_epochs, offset, decay_start_epoch):
		assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
		self.n_epochs = n_epochs
		self.offset = offset
		self.decay_start_epoch = decay_start_epoch

	def step(self, epoch):
		return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		# torch.nn.init.normal(m.weight.data, 0.0, 0.02)
		torch.nn.init.xavier_uniform_(m.weight.data)
	elif classname.find('BatchNorm2d') != -1:
		torch.nn.init.normal(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant(m.bias.data, 0.0)


def soft_threshold(w, th):
	'''
	pytorch soft-sign function
	'''
	with torch.no_grad():
		temp = torch.abs(w) - th
		# print('th:', th)
		# print('temp:', temp.size())
		return torch.sign(w) * nn.functional.relu(temp)


count_ops = 0
num_ids = 0
def get_feature_hook(self, _input, _output):
	global count_ops, num_ids 
	# print('------>>>>>>')
	# print('{}th node, input shape: {}, output shape: {}, input channel: {}, output channel {}'.format(
	# 	num_ids, _input[0].size(2), _output.size(2), _input[0].size(1), _output.size(1)))
	# print(self)
	delta_ops = self.in_channels * self.out_channels * self.kernel_size[0] * self.kernel_size[1] * _output.size(2) * _output.size(3) / self.groups
	count_ops += delta_ops
	# print('ops is {:.6f}M'.format(delta_ops / 1024.  /1024.))
	num_ids += 1
	# print('')

def measure_model(net, H_in, W_in):
	import torch
	import torch.nn as nn
	_input = torch.randn((1, 3, H_in, W_in))
	#_input, net = _input.cpu(), net.cpu()
	hooks = []
	for module in net.named_modules():
		if isinstance(module[1], nn.Conv2d) or isinstance(module[1], nn.ConvTranspose2d):
			# print(module)
			hooks.append(module[1].register_forward_hook(get_feature_hook))

	_out = net(_input)
	global count_ops
	print('count_ops: {:.6f}M'.format(count_ops / 1024. /1024.)) # in Million
	return count_ops


def show_sparsity(model, save_name, model_path=None):
	# load model if necessary:
	if model_path is not None:
		if not os.path.exists(model_path):
			raise Exception("G model path doesn't exist at %s!" % model_path)
		print('Loading generator from %s' % model_path)
		model.load_state_dict(torch.load(model_path))
	
	# get all scaler parameters form the network:
	scaler_list = []
	for m in model.modules():
		if isinstance(m, torch.nn.InstanceNorm2d) and m.weight is not None:
			m_cpu = m.weight.data.cpu().numpy().squeeze()
			# print('m_cpu:', type(m_cpu), m_cpu.shape)
			scaler_list.append(m_cpu)
	all_scaler = np.concatenate(scaler_list, axis=0)
	print('all_scaler:', all_scaler.shape, 'L0 (sum):', np.sum(all_scaler!=0), 'L1 (mean):', np.mean(np.abs(all_scaler)))

	# save npy and plt png:
	# np.save(save_name + '.npy', all_scaler)
	n, bins, patches = plt.hist(all_scaler, 50)
	# print(n)
	plt.savefig(save_name + '.png')
	plt.close()

	return all_scaler

def none_zero_channel_num(model, model_path=None):
	# load model if necessary:
	if model_path is not None:
		if not os.path.exists(model_path):
			raise Exception("G model path doesn't exist at %s!" % model_path)
		print('Loading generator from %s' % model_path)
		model.load_state_dict(torch.load(model_path))
	
	# get all scaler parameters form the network:
	scaler_list = []
	for m in model.modules():
		if isinstance(m, torch.nn.InstanceNorm2d) and m.weight is not None:
			m_cpu = m.weight.data.cpu().numpy().squeeze()
			# print('m_cpu:', type(m_cpu), m_cpu.shape)
			scaler_list.append(m_cpu)
	all_scaler = np.concatenate(scaler_list, axis=0)
	l0norm = np.sum(all_scaler!=0)
	print('all_scaler:', all_scaler.shape, 'L0 (sum):', l0norm, 'L1 (mean):', np.mean(np.abs(all_scaler)))

	return l0norm


def create_dir(_path):
	if not os.path.exists(_path):
		os.makedirs(_path)

def fourD2threeD(batch, n_row=10):
	'''
	Convert a batch of images (N,W,H,C) to a single big image (W*n, H*m, C)
	Input:
		batch: type=ndarray, shape=(N,W,H,C)
	Return:
		rows: type=ndarray, shape=(W*n, H*m, C)
	'''
	N = batch.shape[0]
	img_list = np.split(batch, N)
	for i, img in enumerate(img_list):
		img_list[i] = img.squeeze(axis=0)
	one_row = np.concatenate(img_list, axis=1)
	# print('one_row:', one_row.shape)
	row_list = np.split(one_row, n_row, axis=1)
	rows = np.concatenate(row_list, axis=0)
	return rows



def layer_param_num(model, param_name=['weight']):
	count_res = {}
	for name, W in model.named_parameters():
		if name.strip().split(".")[-1] in param_name and name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
			# W_nz = torch.nonzero(W.data)
			W_nz = torch.flatten(W.data)
			if W_nz.dim() > 0:
				count_res[name] = W_nz.shape[0]
	return count_res


def model_param_num(model, param_name=['weight']):
	'''
	Find parameter numbers in the model. in Million.
	'''
	layer_size_dict = layer_param_num(model, param_name)
	return sum(layer_size_dict.values()) / 1024 / 1024

