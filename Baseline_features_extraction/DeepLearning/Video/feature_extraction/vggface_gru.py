import tensorflow as tf


class VGGFace(object):

    def __init__(self,batch_size):
        self.params = None
        self.batch_size = batch_size
        self.vars = []
        self.layers = []
        self.names = []
        # (1): nn.SpatialConvolutionMM(3 -> 64, 3x3, 1,1, 1,1)
        self.layers.append(('conv','1',3,3,3,64))
        # (3): nn.SpatialConvolutionMM(64 -> 64, 3x3, 1,1, 1,1)
        self.layers.append(('conv','3',3,3,64,64))
        # (5): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (6): nn.SpatialConvolutionMM(64 -> 128, 3x3, 1,1, 1,1)
        self.layers.append(('conv','6',3,3,64,128))
        # (8): nn.SpatialConvolutionMM(128 -> 128, 3x3, 1,1, 1,1)
        self.layers.append(('conv','8',3,3,128,128))
        # (10): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (11): nn.SpatialConvolutionMM(128 -> 256, 3x3, 1,1, 1,1)
        self.layers.append(('conv','11',3,3,128,256))
        # (13): nn.SpatialConvolutionMM(256 -> 256, 3x3, 1,1, 1,1)
        self.layers.append(('conv','13',3,3,256,256))
        # (15): nn.SpatialConvolutionMM(256 -> 256, 3x3, 1,1, 1,1)
        self.layers.append(('conv','15',3,3,256,256))
        # (17): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (18): nn.SpatialConvolutionMM(256 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','18',3,3,256,512))
        # (20): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','20',3,3,512,512))
        # (22): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','22',3,3,512,512))
        # (24): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (25): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','25',3,3,512,512))
        # (27): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','27',3,3,512,512))
        # (29): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','29',3,3,512,512))
        # (31): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (32): nn.View
        self.layers.append(('linear','33',4096,True))
        # (34): nn.ReLU
        # (35): nn.Dropout(0.500000)
        # (36): nn.Linear(4096 -> 4096)
        self.layers.append(('linear2','36',4096,True))
        # (37): nn.ReLU
        # (39): nn.Linear(4096 -> 2)
        self.layers.append(('linear3','39',2,False))


    def get_unique_name_(self, prefix):
        id = sum(t.startswith(prefix) for t,_,_ in self.vars)+1
        return '%s_%d'%(prefix, id)

    def add_(self, name, var,layer):
        self.vars.append((name, var,layer))

    def get_output(self):
        return self.vars[-1][1]

    def get_fc(self):
        return self.vars[-3][1]

    def make_var(self, name, shape,trainable):
        return tf.get_variable(name, shape,trainable=trainable)

    def setup(self,image_batch):
        self.vars.append(('input',image_batch,['input']))
        for layer in self.layers:
            name = self.get_unique_name_(layer[0])
            if layer[0] == 'conv':
                with tf.variable_scope(name) as scope:
                    h, w, c_i, c_o = layer[2],layer[3],layer[4],layer[5]
                    kernel = self.make_var('weights', shape=[h, w, c_i, c_o],trainable=True)
                    conv = tf.nn.conv2d(self.get_output(), kernel, [1]*4, padding='SAME')
                    biases = self.make_var('biases', [c_o],trainable=True)
                    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
                    relu = tf.nn.relu(bias, name=scope.name)
                    self.add_(name, relu,layer)
            elif layer[0] == 'pool':
                size,size,stride,stride = layer[1],layer[2],layer[3],layer[4]
                pool = tf.nn.max_pool(self.get_output(),
                                      ksize=[1, size, size, 1],
                                      strides=[1, stride, stride, 1],
                                      padding='SAME',
                                      name=name)
                self.add_(name, pool,layer)
            elif layer[0] == 'linear':

                num_out = layer[2]
                relu = layer[3]

                with tf.variable_scope(name) as scope:
                    input = self.get_output()
                    input_shape = input.get_shape()
                    if input_shape.ndims==4:
                        dim = 1
                        for d in input_shape[1:].as_list():
                            dim *= d
                        feed_in = tf.reshape(input, [self.batch_size, dim])
                    else:
                        feed_in, dim = (input, int(input_shape[-1]))
                    weights = self.make_var('weights', shape=[dim, num_out],trainable=True)
                    biases = self.make_var('biases', [num_out],trainable=True)
                    op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
                    fc = op(feed_in, weights, biases, name=scope.name)
                    print(fc)
                    ##### if you are training, use this
                    #drop = tf.nn.dropout(fc,0.5)
                    #self.add_(name, drop,layer)
                    #####
                    ##### if you are testing, use this instead
                    self.add_(name, fc,layer)
                    #####

            elif layer[0] == 'linear2':
                num_out = layer[2]
                relu = layer[3]

                with tf.variable_scope(name) as scope:
                    input = self.get_output()
                    input_shape = input.get_shape()
                    if input_shape.ndims==4:
                        dim = 1
                        for d in input_shape[1:].as_list():
                            dim *= d
                        feed_in = tf.reshape(input, [self.batch_size, dim])
                    else:
                        feed_in, dim = (input, int(input_shape[-1]))

                    weights = self.make_var('weights', shape=[dim, num_out],trainable=True)
                    biases = self.make_var('biases', [num_out],trainable=True)
                    op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
                    fc = op(feed_in, weights, biases, name=scope.name)
                    self.add_(name, fc,layer)
            elif layer[0] == 'linear3':

                num_out = layer[2]
                relu = layer[3]

                with tf.variable_scope(name) as scope:
                    input = self.get_output()
                    input_shape = input.get_shape()
                    if input_shape.ndims==4:
                        dim = 1
                        for d in input_shape[1:].as_list():
                            dim *= d
                        feed_in = tf.reshape(input, [self.batch_size, dim])
                    else:
                        feed_in, dim = (input, int(input_shape[-1]))
                    weights = self.make_var('weights', shape=[dim, num_out],trainable=True)
                    biases = self.make_var('biases', [num_out],trainable=True)
                    op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
                    fc = op(feed_in, weights, biases, name=scope.name)

                    self.add_(name, fc,layer)