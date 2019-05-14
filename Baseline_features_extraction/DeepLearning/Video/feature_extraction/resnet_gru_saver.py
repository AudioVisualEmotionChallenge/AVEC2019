from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib import slim
import tensorflow as tf

sequence_length = 80
batch_size = 1
images_batch = tf.placeholder(tf.float32, [80, 96, 96, 3])

with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    net, end_point = resnet_v1.resnet_v1_50(inputs=images_batch, is_training=False, num_classes=None)

saver_resnet = tf.train.Saver(slim.get_model_variables())
# with tf.variable_scope('rnn') as scope:
    # cnn = tf.reshape(net, [batch_size, sequence_length, -1])
    # cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(128) for _ in range(2)])
    # outputs, _ = tf.nn.dynamic_rnn(cell, cnn, dtype=tf.float32)
    # outputs = tf.reshape(outputs, (batch_size * sequence_length, 128))

    # weights_initializer = tf.truncated_normal_initializer(
        # stddev=0.01)
    # weights = tf.get_variable('weights_output',
                              # shape=[128, 2],
                              # initializer=weights_initializer,
                              # trainable=True)
    # biases = tf.get_variable('biases_output',
                             # shape=[2],
                             # initializer=tf.zeros_initializer, trainable=True)

    # prediction = tf.nn.xw_plus_b(outputs, weights, biases)
    # valence_val = prediction[:, 0]
    # arousal_val = prediction[:, 1]

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, '/home/psxss8/openface/OpenFace/build/bin/Affwild_models/resnet_v1_50.ckpt')
saver_resnet.save(sess, '/home/psxss8/openface/OpenFace/build/bin/Affwild_models/standard_ResNet/model.cktp')

