"""
Defines models
"""

import numpy

import torch
import torchtext
import torchvision.models
import torch.backends.cudnn

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_

from collections import OrderedDict

device = torch.device('cuda')


#******************** UNIMODAL REGRESSOR SEQUENCE ********************#
class UnimodalRegressorSequence(torch.nn.Module):
	"""
	RNN-based unimodal regressor model
	"""
	def __init__(self, opt):
		super(UnimodalRegressorSequence, self).__init__()

		self.feature_dim = opt.feature_dim
		self.rnn_layer_dim = opt.rnn_layer_dim
		self.output_dim = opt.class_num

		self.rnn_layer_num = opt.rnn_layer_num
		self.bidirectional = opt.bidirectional
		self.dropout_rate = opt.dropout_rate

		# Layer parameters
		self.rnn = torch.nn.GRU(input_size=self.feature_dim,
								hidden_size=self.rnn_layer_dim,
								num_layers=self.rnn_layer_num,
								bidirectional=self.bidirectional,
								batch_first=True)

		#self.linear1 = torch.nn.Linear(self.rnn_layer_dim, opt.hidden_layer_dim)
		#self.relu = torch.nn.ReLU()
		self.dropout = torch.nn.Dropout(self.dropout_rate)
		self.linear2 = torch.nn.Linear(opt.hidden_layer_dim, self.output_dim)
		#self.activation = torch.nn.Sigmoid()
		#self.activation = torch.nn.Hardtanh(min_val=0.0, max_val=1.0)

		# Model parameters
		self.optim_params = []
		self.optim_params += list(self.rnn.parameters())
		#self.optim_params += list(self.linear1.parameters())
		self.optim_params += list(self.linear2.parameters())

	def forward_once(self, seq, length):
		packed = pack_padded_sequence(seq, length, batch_first=True)
		rnn_out, _ = self.rnn(packed)
		padded = pad_packed_sequence(rnn_out, batch_first=True)
		I = torch.LongTensor(length).view(-1, 1, 1)
		I = Variable(I.expand(seq.size(0), 1, self.rnn_layer_dim) - 1).cuda()

		x = torch.gather(padded[0], 1, I).squeeze(1)
		#x = self.linear1(x)
		#x = self.relu(x)
		x = self.dropout(x)
		x = self.linear2(x)
		#y = self.activation(x)

		return x

	def forward(self, seq, length):
		idx = sorted(range(len(length)), key=length.__getitem__, reverse=True)
		ridx = sorted(range(len(length)), key=idx.__getitem__)
		
		y = self.forward_once(seq[idx,:], sorted(length, reverse=True))
		y = y[ridx,:]

		return y
