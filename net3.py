##################################################
# Author: Peizhi Yan                             #
# Affiliation: Lakehead University               #
# Personal Website: https://PeizhiYan.github.io  #
# Date: Jan. 23, 2019                            #
##################################################

import tensorflow as tf

"""fully connected layer"""
def dense(input_tensor, input_units, units, device, name_scope):
    with tf.device(device):
        with tf.name_scope(name_scope):
            W = tf.Variable(tf.random_normal((input_units, units)))
            b = tf.Variable(tf.random_normal((units,)))
            return tf.nn.relu(tf.matmul(input_tensor, W) + b)

"""transpose convolution layer in 2D (Note well!! Instead of using this function, I choose to use 
tf.image.resize_images() )"""
def t_conv2d(input_tensor, filter_size, output_size, in_channels, out_channels, upsample, device, name_scope):
    '''
    ; input_tensor: the input tensor
    ; filter_size: the size (size == width == height) of the filter
    ; output_size: the output size (size == width == height) of feature map
    ; in_channels: the number of channels in input_tensor
    ; out_channels: the number fo channels in output tensor (in other word, the number of feature maps)
    ; upsample: True/False. True: output size == 2 times the size of input; False: output size == input size 
    ; device: e.g., '/cpu:0' or '/gpu:0'
    ; name_scope: the name scope in computational graph
    '''
    with tf.device(device):
        with tf.name_scope(name_scope):
            strides = [1, 1, 1, 1]
            if upsample == True:
                strides = [1, 2, 2, 1] # 2X upsampling
            dyn_input_shape = tf.shape(input_tensor)
            batch_size = dyn_input_shape[0]
            out_shape = tf.stack([batch_size, output_size, output_size, out_channels])
            filter_shape = [filter_size, filter_size, out_channels, in_channels]
            w = tf.get_variable(name=name_scope+'weights', shape=filter_shape)
            h1 = tf.nn.conv2d_transpose(input_tensor, w, out_shape, strides, padding='SAME')
            return h1

"""convolution layer in 2D"""
def conv2d(input_tensor, filter_size, in_channels, out_channels, device, name_scope, activation=True):
    '''
    ; input_tensor: the input tensor
    ; filter_size: the size (size == width == height) of the filter
    ; in_channels: the number of channels in input_tensor
    ; out_channels: the number fo channels in output tensor (in other word, the number of feature maps)
    ; device: e.g., '/cpu:0' or '/gpu:0'
    ; name_scope: the name scope in computational graph
    '''
    with tf.device(device):
        with tf.name_scope(name_scope):
            kernel = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 
                                                 dtype=tf.float32, stddev=1e-1), name='weights', trainable=True)
            conv = tf.nn.conv2d(input_tensor, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[out_channels], dtype=tf.float32), trainable=True, name='biases')
            conv_biased = tf.nn.bias_add(conv, biases)
            if activation == False:
                return conv_biased # no ReLU activation
            relu_conv = tf.nn.relu(conv_biased)
            return relu_conv

"""2D max pooling layer"""
def max_pool(input_tensor, device, name_scope):
    '''
    ; input_tensor: the input tensor
    ; device: e.g., '/cpu:0' or '/gpu:0'
    ; name_scope: the name scope in computational graph
    '''
    with tf.device(device):
        with tf.name_scope(name_scope):
            return tf.nn.max_pool(input_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

"""build the deep auto-encoder network"""
def net(X, device, manual_feature):
    with tf.device(device):
        # zero-mean input  [-1,224,224,3]
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68/255, 116.779/255, 103.939/255], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean') # mean from ImageNet
            #IMAGES = X - mean
            IMAGES = X
        
        """Encoding (convolution and max-pooling)"""
        
        # conv1  output: [-1,224,224,64]
        conv1 = conv2d(input_tensor=IMAGES, filter_size=3, in_channels=3, out_channels=64, device=device, name_scope='conv1')
    
        # pool1  output: [-1,112,112,64]
        pool1 = max_pool(input_tensor=conv1, device=device, name_scope='pool1')
        
        # conv2  output: [-1,112,112,128]
        conv2 = conv2d(input_tensor=pool1, filter_size=3, in_channels=64, out_channels=128, device=device, name_scope='conv2')
        
        # pool2    output: [-1,56,56,128]
        pool2 = max_pool(input_tensor=conv2, device=device, name_scope='pool2')

        # conv3  output: [-1,56,56,256]
        conv3 = conv2d(input_tensor=pool2, filter_size=3, in_channels=128, out_channels=256, device=device, name_scope='conv3')

        # pool3    output: [-1,28,28,256]
        pool3 = max_pool(input_tensor=conv3, device=device, name_scope='pool3')

        # conv4  output: [-1,28,28,512]
        conv4 = conv2d(input_tensor=pool3, filter_size=3, in_channels=256, out_channels=512, device=device, name_scope='conv4')

        # pool4    output: [-1,14,14,512]
        pool4 = max_pool(input_tensor=conv4, device=device, name_scope='pool4')
        
        # conv5  output: [-1,14,14,256]
        conv5 = conv2d(input_tensor=pool4, filter_size=3, in_channels=512, out_channels=256, device=device, name_scope='conv5')
        
        # conv6  output: [-1,14,14,128]
        conv6 = conv2d(input_tensor=conv5, filter_size=3, in_channels=256, out_channels=128, device=device, name_scope='conv6')
        
        # conv7  output: [-1,14,14,64]
        conv7 = conv2d(input_tensor=conv6, filter_size=3, in_channels=128, out_channels=64, device=device, name_scope='conv7')
        
        # pool5  output: [-1,7,7,64]
        pool5 = max_pool(input_tensor=conv7, device=device, name_scope='pool5')
        
        encoder = pool5
        
        
        """Decoding (up-sampling and convolution)"""
        
        # up-sampling 1 (nearest neighbor interpolation)  output: [-1,14,14,64]
        up1 = tf.image.resize_images(pool5, size=(14,14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        # conv8_1 output: [-1,14,14,128]
        conv8_1 = conv2d(input_tensor=up1, filter_size=3, in_channels=64, out_channels=128, device=device, name_scope='conv8_1')
        
        # conv8_2 output: [-1,14,14,256]
        conv8_2 = conv2d(input_tensor=conv8_1, filter_size=3, in_channels=128, out_channels=256, device=device, name_scope='conv8_2')
        
        # up-sampling 2 (nearest neighbor interpolation)  output: [-1,28,28,256]
        up2 = tf.image.resize_images(conv8_2, size=(28,28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        # conv9_1 output: [-1,28,28,512]
        conv9_1 = conv2d(input_tensor=up2, filter_size=3, in_channels=256, out_channels=512, device=device, name_scope='conv9_1')
        
        # up-sampling 3 (nearest neighbor interpolation)  output: [-1,56,56,512]
        up3 = tf.image.resize_images(conv9_1, size=(56,56), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        # conv10_1 output: [-1,56,56,512]
        conv10_1 = conv2d(input_tensor=up3, filter_size=3, in_channels=512, out_channels=512, device=device, name_scope='conv10_1')
        
        # up-sampling 4 (nearest neighbor interpolation)  output: [-1,112,112,512]
        up4 = tf.image.resize_images(conv10_1, size=(112,112), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        # conv11_1 output: [-1,112,112,512]
        conv11_1 = conv2d(input_tensor=up4, filter_size=3, in_channels=512, out_channels=512, device=device, name_scope='conv11_1')
        
        # conv11_2 output: [-1,112,112,512]
        conv11_2 = conv2d(input_tensor=conv11_1, filter_size=3, in_channels=512, out_channels=512, device=device, name_scope='conv11_2')
        
        # up-sampling 5 (nearest neighbor interpolation)  output: [-1,224,224,512]
        up5 = tf.image.resize_images(conv11_2, size=(224,224), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        # conv12_1 output: [-1,224,224,512]
        conv12_1 = conv2d(input_tensor=up5, filter_size=3, in_channels=512, out_channels=512, device=device, name_scope='conv12_1')
        
        # conv12_2 output: [-1,224,224,512]
        conv12_2 = conv2d(input_tensor=conv12_1, filter_size=3, in_channels=512, out_channels=512, device=device, name_scope='conv12_2')
        
        # conv12_3 output: [-1,224,224,3]
        conv12_3 = conv2d(input_tensor=conv12_2, filter_size=3, in_channels=512, out_channels=3,
                         device=device, name_scope='conv12_3',activation = False) # no activation in this layer!!
        
        """decoder"""
        decoder = tf.nn.sigmoid(conv12_3) # through sigmoid to get reconstructed image  (pixel range: 0.0~1.0)
        
        """loss"""
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=IMAGES, logits=conv12_3) # cross-entropy loss
        loss = tf.reduce_mean(loss)
        
        return encoder, decoder, loss