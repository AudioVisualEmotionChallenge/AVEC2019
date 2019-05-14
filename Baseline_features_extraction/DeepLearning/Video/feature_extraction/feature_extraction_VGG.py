import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import vggface_gru
import scipy.io as io
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--weights', default="model.ckpt-975", type=str) # define the model path
parser.add_argument('--weight_dir', default='./Affwild_models/VGG/', type=str) # define the model path
parser.add_argument('--input_file', default='video_T_01.csv', type=str) # define the input image path
parser.add_argument('--save_file', default='vggface.mat', type=str) # define the path to save extracted features
args = parser.parse_args()



class Detector(object):
    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)
        self.fc1 = self.net.get_fc()

    def detect(self, img):
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (96, 96))

        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, 96, 96, 3))
        net_output = self.sess.run(self.fc1,
                                   feed_dict={self.net.vars[0][1]: inputs})
        return net_output

		
		
def main():

    os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    network = vggface_gru.VGGFace(1)
    image_batch= np.zeros((1,96,96,3),dtype='float32')
    image_tensor=tf.convert_to_tensor(image_batch)
    network.setup(image_tensor)
    weight_file = os.path.join(args.weight_dir, args.weights)
    detector = Detector(network, weight_file)

    #files = pd.read_csv('video_T_01.csv')
    files = pd.read_csv(args.input_file)
    files = files.values
    feature_list = []
    for file_path in tqdm(files):
        file_path = file_path[0].strip()
        image = cv2.imread(file_path)
        feature = detector.detect(image)
        feature_list.append(feature)
    feature_list = np.concatenate(feature_list, axis=0)
    io.savemat(args.save_file, {'feature': feature_list})

    # lines = open("list_left.txt", "r").readlines()
    # path='./img/'
    # save_path=os.path.join(path,"single")
    # lines=os.listdir(path)
    # lines=[os.path.join(path,e) for e in lines]

    # if not os.path.exists(im_name):
    #     print("Error")
    # image = cv2.imread(im_name)
    # rows, cols, _ = image.shape
    # result = detector.detect(image)
    # print(result.shape)
    # io.savemat(save_path,{'features':result})


if __name__ == '__main__':
    main()