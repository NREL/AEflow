import tensorflow as tf
import numpy as np

class AEflow:
    def __init__(self, x, status='full', reuse=False):
        '''
        Inputs:
               x: Input to the network. Depends on the given status
          Status: Which mode to run the model in
           Options:
              'full': Input the full data field and output the compressed then reconstructed data
          'compress': Input the full data field and output the compressed data
       'reconstruct': Input the compressed data field and output the reconstructed data

        Variables:
                  y: Output of the model. Depends on the given status
           var_list: List of variables to determine how much of the model to load
               loss: Loss function for training
        '''
        if x is None: return

        status = status.lower()
        assert status in ['full', 'compress', 'reconstruct'], 'Error in autoencoder status.'

        if status in ['full', 'compress']:
            self.y = self.__encoder__(x, reuse)
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
        else:
            self.y = x
            self.var_list = []

        if status in ['full', 'reconstruct']:
            self.y = self.__decoder__(self.y, reuse)
            self.var_list = (self.var_list + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder'))

        if status == 'full':
            self.loss = tf.reduce_mean((self.y - x)**2)
        else:
            self.loss = None


    def __conv_layer_3d__(self, x, filter_shape, stride):
        filter_ = tf.get_variable(name='weight', shape=filter_shape, dtype=tf.float64,
                                    initializer=tf.contrib.layers.xavier_initializer(), trainable=True)

        return tf.nn.conv3d(input=x, filter=filter_, strides=[1, stride, stride, stride, 1],
                              padding='SAME')


    def __deconv_layer_3d__(self, x, filter_shape, output_shape, stride):
        filter_ = tf.get_variable(name='weight', shape=filter_shape, dtype=tf.float64,
                                    initializer=tf.contrib.layers.xavier_initializer(), trainable=True)

        return tf.nn.conv3d_transpose(value=x, filter=filter_, output_shape=output_shape,
                                        strides=[1, stride, stride, stride, 1])


    def __encoder__(self, x, reuse):
        with tf.variable_scope('encoder', reuse=reuse):

            C = 4
            with tf.variable_scope('conv_1'):
                x = self.__conv_layer_3d__(x, [5, 5, 5, x.get_shape()[-1], C], 1)
                x = tf.nn.leaky_relu(x)

            skip_connection = x
            
            for i in range(12):
                B_skip_connection = x

                with tf.variable_scope('residual_block_{}a'.format(i)):
                    x = self.__conv_layer_3d__(x, [5, 5, 5, x.get_shape()[-1], C], 1)
                    x = tf.nn.relu(x)

                with tf.variable_scope('residual_block_{}b'.format(i)):
                    x = self.__conv_layer_3d__(x, [5, 5, 5, C, C], 1)

                x = tf.add(x, B_skip_connection)

            with tf.variable_scope('conv_2'):
                x = self.__conv_layer_3d__(x, [5, 5, 5, C, C], 1)
                x = tf.add(x, skip_connection)
            
            C_mult = [2, 2, 1]
            for i in range(3):
                C = int(x.get_shape()[-1])

                with tf.variable_scope('compress_block_{}a'.format(i)):
                    x = self.__conv_layer_3d__(x, [5, 5, 5, C, C], 2)
                    x = tf.nn.leaky_relu(x, alpha=0.2)

                with tf.variable_scope('compress_block_{}b'.format(i)):
                    x = self.__conv_layer_3d__(x, [5, 5, 5, C, C_mult[i]*C], 1)
                    x = tf.nn.leaky_relu(x, alpha=0.2)
            
            with tf.variable_scope('conv_out'):
                C = int(x.get_shape()[-1])
                x = self.__conv_layer_3d__(x, [5, 5, 5, C, int(C/2)], 1)

            return x
            

    def __decoder__(self, x, reuse):
        with tf.variable_scope('decoder', reuse=reuse):
            with tf.variable_scope('conv_1'):
                C = int(x.get_shape()[-1])
                x = self.__conv_layer_3d__(x, [5, 5, 5, C, 2*C], 1)

            C_div = [1, 2, 2]
            for i in range(3):
                N, h, w, d, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], x.get_shape()[4]
                C_over_div = int(int(C)/C_div[i])

                with tf.variable_scope('decompress_block_{}a'.format(i)):
                    x = self.__conv_layer_3d__(x, [5, 5, 5, C, C_over_div], 1)
                    x = tf.nn.leaky_relu(x, alpha=0.2)

                with tf.variable_scope('decompress_block_{}b'.format(i)):
                    x = self.__deconv_layer_3d__(x, [5, 5, 5, C_over_div, C_over_div], [N, 2*h, 2*w, 2*d, C_over_div], 2)
                    x = tf.nn.leaky_relu(x, alpha=0.2)

            skip_connection = x

            C = 4
            for i in range(12):
                B_skip_connection = x

                with tf.variable_scope('residual_block_{}a'.format(i)):
                    x = self.__conv_layer_3d__(x, [5, 5, 5, x.get_shape()[-1], C], 1)
                    x = tf.nn.relu(x)

                with tf.variable_scope('residual_block_{}b'.format(i)):
                    x = self.__conv_layer_3d__(x, [5, 5, 5, C, C], 1)

                x = tf.add(x, B_skip_connection)

            with tf.variable_scope('conv_2'):
                x = self.__conv_layer_3d__(x, [5, 5, 5, C, C], 1)
                x = tf.add(x, skip_connection)

            with tf.variable_scope('conv_out'):
                x = self.__conv_layer_3d__(x, [5, 5, 5, C, 1], 1)

        return x




