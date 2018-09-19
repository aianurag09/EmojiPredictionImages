import cv2
import numpy as np
import copy
import cPickle

from scipy.misc import imshow, imread, imresize

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, add
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
from keras.engine import Layer, InputSpec
from keras import backend as K

import sys
import os
sys.setrecursionlimit(3000)

class Scale(Layer):
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name='%s_gamma'%self.name)
        self.beta = K.variable(self.beta_init(shape), name='%s_beta'%self.name)
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def identity_block(input_tensor, kernel_size, filters, stage, block):
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x



class model(object):

    def __init__(self):
        self.eps = 1.1e-5
        self.build_model()


    def build_model(self):

        global bn_axis
        if K.image_dim_ordering() == 'tf':
            bn_axis = 3
            img_input = Input(shape=(224, 224, 3), name='data')
        else:
            bn_axis = 1
            img_input = Input(shape=(3, 224, 224), name='data')
                
        x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
        x = BatchNormalization(epsilon=self.eps, axis=bn_axis, name='bn_conv1')(x)
        x = Scale(axis=bn_axis, name='scale_conv1')(x)
        x = Activation('relu', name='conv1_relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        for i in range(1,8):
            x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        for i in range(1,36):
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
        x_fc = Flatten()(x_fc)
        x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

        self.model = Model(img_input, x_fc)


    def set_weights(self, weights_path):
        if not os.path.exists(weights_path):
            raise IOError("{} does not exist".format(weights_path))
        else:
            self.model.load_weights(weights_path, by_name=True)            


    def compile(self):
        self.sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=self.sgd, loss='categorical_crossentropy', metrics=['accuracy'])


    def predict(self, test_image):
        if os.path.exists(test_image):
            img = self.imread(test_image)
            return self.model.predict(img)
        else:
            raise IOError("{} image does not exist".format(test_image))

    def mean_subtract(self, img):
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68

        return img

    def imread(self, path):
        img = cv2.resize(cv2.imread(path), (224, 224)).astype(np.float32)
        img = self.mean_subtract(img)

        if K.image_dim_ordering() == 'th':
            # Transpose image dimensions (Theano uses the channels as the 1st dimension)
            img = img.transpose((2,0,1))
        img = np.expand_dims(img, axis=0)

        return img
