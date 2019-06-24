"""
Trains and validates models
"""

import os
import time
import logging
import argparse

import pickle
import pandas as pd
import numpy as np
import shutil

from sklearn.metrics import mean_squared_error
from math import sqrt
import torch

from create_datasets import get_loaders_unimodal_regressor_sequence_dataset
from define_models import UnimodalRegressorSequence

# Reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(0)

def main():
	parser = argparse.ArgumentParser()

	torch.cuda.set_device('cuda')

	# Names, paths, logs
	parser.add_argument('--dataset_path', default='./data', help='path to dataset')
	parser.add_argument('--dataset_file_path', default='all_subjects_scaled.csv', help='path to dataset file')
	parser.add_argument('--logger_path', default='./checkpoints', help='path to log')
	parser.add_argument('--out_path', default='./out', help='Directory path for predictions')
	parser.add_argument('--logger_name', default='logging.log', help='dcaps-discloure.log|dcaps-sentiment.log')

	# Data parameters
	parser.add_argument('--feature_dim', default=39, type=int, help='dimensionality of the features (mfcc = 39|eGeMAPS = 23|AUpose = 49|ResNet = 2048')
	parser.add_argument('--feature_type', default='mfcc', help='mfcc|eGeMAPS|AUpose|ResNet|VGG|DS_densenet')
	parser.add_argument('--workers_num', default=4, type=int, help='number of workers for data loading')
	parser.add_argument('--class_num', default=1, type=int, help='number of classes')
	parser.add_argument('--max_sequence_length', default=120000, type=int, help='maximum length of feature sequences')
	parser.add_argument('--modality', default='speech', help='speech|vision')

	# Model parameters
	parser.add_argument('--model_type', default='unimodal-regressor-sequence', help='unimodal-regressor-sequence')
	parser.add_argument('--rnn_layer_dim', default=64, help='dimensionality of RNN layers')
	parser.add_argument('--hidden_layer_dim', default=64, help='dimensionality of Hidden layers')
	parser.add_argument('--bidirectional', default=False, help='bidirectional RNN (embedding_dim will be halved)')
	parser.add_argument('--rnn_layer_num', default=1, help='number of RNN layers')
	parser.add_argument('--dropout_rate', default=.2, help='dropout rate')

	# Training and optimization
	parser.add_argument('--epochs_num', default=30, help='number of training epochs')
	parser.add_argument('--batch_size', default=15, help='size of a mini-batch')
	parser.add_argument('--weight_decay', default=.0, help='decay (l2 norm) for the optimizer weights')
	parser.add_argument('--learning_rate', default=.0001, help='MFCC=0.01')
	parser.add_argument('--learning_rate_num', default=10000, help='number of epochs for the update of the learning rate')

	opt = parser.parse_args()

	train_data = pd.read_csv(os.path.join(opt.dataset_path, 'train_split.csv'), header=0)
	train_ids = train_data['Participant_ID'].tolist()

	val_data = pd.read_csv(os.path.join(opt.dataset_path, 'dev_split.csv'), header=0)
	val_ids = val_data['Participant_ID'].tolist()

	test_data = pd.read_csv(os.path.join(opt.dataset_path, 'test_split.csv'), header=0)
	test_ids = test_data['Participant_ID'].tolist()

	ids = {}

	if not os.path.exists(opt.logger_path):
		os.makedirs(opt.logger_path)
	if not os.path.exists(opt.out_path):
		os.makedirs(os.path.join(opt.out_path, 'predictions', 'val'))
		os.makedirs(os.path.join(opt.out_path, 'predictions', 'test'))
		os.makedirs(os.path.join(opt.out_path, 'checkpoints'))

	set_logger(os.path.join(opt.logger_path, opt.logger_name))
	ids['train'] = train_ids
	ids['test'] = test_ids
	ids['val'] = val_ids

	set_logger(os.path.join(opt.logger_path, opt.logger_name))

# Data loaders
	if opt.modality=='speech' or opt.modality=='vision':
		train_loader, val_loader, test_loader = get_loaders_unimodal_regressor_sequence_dataset(ids=ids,opt=opt)
	else:
		print('Data loader is not implemented for ', opt.modality)

	# Model and optimizer
	if opt.modality=='speech' or opt.modality=='vision':
		if opt.model_type=='unimodal-regressor-sequence':
			model = UnimodalRegressorSequence(opt=opt).cuda()
			optimizer = torch.optim.Adam(model.optim_params, lr=float(opt.learning_rate), weight_decay=opt.weight_decay, amsgrad=True)
		else:
			print('Model is not implemented for ', opt.modality, '->', opt.model_type)

	else:
		print('Model is not implemented for ', opt.modality)

	# Train and validate
	print(model)
	best_ccc = -1

	best_file_name = os.path.join(opt.out_path, 'checkpoints', opt.feature_type+'_best_model.pth.tar')

	for epoch in range(int(opt.epochs_num)):

		if opt.model_type=='unimodal-regressor-sequence':
			train_loss, train_ccc = train_unimodal_regressor_sequence(train_loader, model, optimizer, opt)

			val_ccc, val_predictions, _ = validate_unimodal_regressor_sequence(val_loader, model)
			test_predictions  = test_unimodal_regressor_sequence(test_loader, model)

			checkpoint_file_name = os.path.join(opt.out_path, 'checkpoints', opt.feature_type+'_epoch'+str(epoch+1)+'.pth.tar')

			state = {'epoch': epoch+1, 'model': model.state_dict(), 'opt': opt}
			torch.save(state, checkpoint_file_name)

			is_best = val_ccc > best_ccc

			if is_best:
				shutil.copyfile(checkpoint_file_name, best_file_name)
				best_ccc = val_ccc
				best_val_predictions = val_predictions
				best_test_predictions = test_predictions

			msg = 'epoch: {0:.0f}'.format(epoch+1) + ' loss: {0:.5f}'.format(train_loss) + ' train_score: {0:.5f} '.format(train_ccc) + ' val_score: {0:.5f} '.format(val_ccc)
			logging.log(msg=msg, level=logging.DEBUG)

		else:
			print('Train and validate is not implemented for ', opt.model_type)

	msg = "Best validation score: {0:.5f}".format(best_ccc)
	logging.log(msg = msg, level=logging.DEBUG)

	val_labels = val_data['PHQ_Score'].tolist()
	best_val_predictions = best_val_predictions * 25
	best_test_predictions = best_test_predictions * 25

	val_rmse = sqrt(mean_squared_error(val_labels, best_val_predictions))

	logging.log(msg='Val RMSE: '+str(val_rmse), level=logging.DEBUG)

	np.save(os.path.join(opt.out_path, 'predictions', 'val', opt.feature_type+'.npy'), best_val_predictions)
	np.save(os.path.join(opt.out_path, 'predictions', 'test', opt.feature_type+'.npy'), best_test_predictions)

def train_unimodal_regressor_sequence(train_loader, model, optimizer, opt):
	running_loss = 0.

	predictions_corr = list()
	labels_corr = list()

	for i, train_data in enumerate(train_loader):
		features, lengths, labels, _ = train_data
		optimizer.zero_grad()

		features = features.cuda()
		labels = labels.cuda()

		predictions = model.forward(features, lengths)
		labels = labels.view(predictions.size()[0], -1)

		loss = custom_loss(predictions, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()

		with torch.no_grad():
			predictions_tmp = predictions.cpu().numpy().reshape(1,-1).tolist()[0]
			labels_tmp = labels.cpu().numpy().reshape(1,-1).tolist()[0]

			predictions_corr += predictions_tmp
			labels_corr += labels_tmp

	train_ccc = ccc_score(np.array(labels_corr), np.array(predictions_corr))
	return running_loss / len(train_loader), train_ccc

def validate_unimodal_regressor_sequence(val_loader, model):
	with torch.no_grad():
		predictions_corr = np.empty((0, 1))
		labels_corr = np.empty((0, 1))

		for i, val_data in enumerate(val_loader):
			features, lengths, labels, indx = val_data
			features = features.cuda()

			predictions = model.forward(features, lengths)

			predictions = predictions.cpu().numpy()
			labels = np.expand_dims(labels.cpu().numpy(), axis=1)

			predictions_corr = np.append(predictions_corr, predictions[indx], axis=0)
			labels_corr = np.append(labels_corr, labels[indx], axis=0)

		labels_corr = labels_corr.reshape(1,-1)[0]
		predictions_corr = predictions_corr.reshape(1,-1)[0]
		ccc = ccc_score(labels_corr,predictions_corr)
	return ccc, predictions_corr, labels_corr

def test_unimodal_regressor_sequence(test_loader, model):

	with torch.no_grad():
		predictions_corr = np.empty((0, 1))

		for i, val_data in enumerate(test_loader):
			features, lengths, indx = val_data

			features = features.cuda()
			predictions = model.forward(features, lengths)
			predictions = predictions.cpu().numpy()
			predictions_corr = np.append(predictions_corr, predictions[indx], axis=0)

		predictions_corr = predictions_corr.reshape(1,-1)[0]

	return predictions_corr



def set_logger(log_path):
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)

	if not logger.handlers:
		file_handler = logging.FileHandler(log_path)
		file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
		logger.addHandler(file_handler)

		# Logging to console
		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(logging.Formatter('%(message)s'))
		logger.addHandler(stream_handler)

def custom_loss(output, target):

	out_mean = torch.mean(output)
	target_mean = torch.mean(target)

	covariance = torch.mean( (output - out_mean) * (target - target_mean) )
	target_var = torch.mean( (target - target_mean)**2)
	out_var = torch.mean( (output - out_mean)**2 )

	ccc = 2.0 * covariance/(target_var + out_var + (target_mean-out_mean)**2 + 1e-10)
	loss_ccc = 1.0 - ccc

	return loss_ccc

def ccc_score(x, y):
	# Computes the metrics CCC
	#  CCC:  Concordance correlation coeffient
	# Input:  x,y: numpy arrays (one-dimensional)
	# Output: CCC

	x_mean = np.nanmean(x)
	y_mean = np.nanmean(y)

	covariance = np.nanmean((x - x_mean) * (y - y_mean))

	x_var = np.nanmean((x - x_mean) ** 2)
	y_var = np.nanmean((y - y_mean) ** 2)

	CCC = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2)

	return CCC

if __name__ == '__main__':
	main()
