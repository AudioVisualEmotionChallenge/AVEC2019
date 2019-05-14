import os
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib import slim
import tensorflow as tf
import argparse
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
import scipy.io as io

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default="model.cktp", type=str) # define the model path
parser.add_argument('--weight_dir', default='./Affwild_models/standard_ResNet/', type=str) # define the model path
parser.add_argument('--input_file', default='video_T_01.csv', type=str) # define the input image path
parser.add_argument('--save_file', default='video_T_01.mat', type=str) # define the path to save extracted features
args = parser.parse_args()


images_batch = tf.placeholder(tf.float32, [1, 96, 96, 3])

with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    net, end_point = resnet_v1.resnet_v1_50(inputs=images_batch, is_training=False, num_classes=None)
net = tf.squeeze(net, [1, 2])
saver = tf.train.Saver()
sess = tf.Session()
weight_file = os.path.join(args.weight_dir, args.weights)
saver.restore(sess, weight_file)  
files = pd.read_csv(args.input_file)
files = files.values
feature_list = []
for file_path in tqdm(files):
    file_path = file_path[0].strip()
    image = cv2.imread(file_path)
    inputs = cv2.resize(image, (96, 96))
    inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
    inputs -= 128.0
    inputs /= 128.0
    feature = sess.run(net, feed_dict={images_batch: [inputs]})
    feature_list.append(feature)

feature_list = np.concatenate(feature_list, axis=0)
io.savemat(args.save_file, {'feature': feature_list})