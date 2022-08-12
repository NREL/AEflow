from email.errors import HeaderMissingRequiredValue
import tensorflow as tf
import numpy as np
import os
import utils

class AEflow1D:
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

        
        self.input = x
        if status in ['full', 'compress']:
            self.y = self.__encoder__(x, reuse)
        else:
            self.y = x

        if status in ['full', 'reconstruct']:
            self.y = self.__decoder__(self.y, reuse)

    def __conv_layer_3d__(self, output_filters, stride, name):
        return tf.keras.layers.Conv3D(filters=output_filters, kernel_size=5, strides=stride, padding='same', name=name, use_bias=False)
        
    def __deconv_layer_3d__(self, output_filters, stride, name):
        return tf.keras.layers.Conv3DTranspose(filters=output_filters, kernel_size=5, strides=stride, padding='same', name=name, use_bias=False)

    def __encoder__(self, x, reuse):
         with tf.compat.v1.variable_scope('encoder', reuse=reuse):
            C = 4
            x = self.__conv_layer_3d__(C, 1, 'e_conv_1')(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

            skip_connection = x

            for i in range(12):
                B_skip_connection = x

                x = self.__conv_layer_3d__(C, 1, 'e_residual_block_{}a'.format(i))(x)
                x = tf.keras.layers.ReLU()(x)

                x = self.__conv_layer_3d__(C, 1, 'e_residual_block_{}b'.format(i))(x)

                x = tf.keras.layers.Add()([x, B_skip_connection])

            x = self.__conv_layer_3d__(C, 1, 'e_conv_2')(x)
            x = tf.keras.layers.Add()([x, skip_connection])
            
            C_mult = [2, 2, 1]
            for i in range(3):
                C = int(x.get_shape()[-1])

                x = self.__conv_layer_3d__(C, 2, 'e_compress_block_{}a'.format(i))(x)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

                x = self.__conv_layer_3d__(C_mult[i]*C, 1, 'e_compress_block_{}b'.format(i))(x)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

            
            C = int(x.get_shape()[-1])
            x = self.__conv_layer_3d__(int(C/2), 1, 'e_conv_out')(x)

            return x
            

    def __decoder__(self, x, reuse):
        with tf.compat.v1.variable_scope('decoder', reuse=reuse):
            C = int(x.get_shape()[-1])
            x = self.__conv_layer_3d__(2*C, 1, name='d_conv_1')(x)

            C_div = [1, 2, 2]
            for i in range(3):
                C = x.get_shape()[4]
                C_over_div = int(int(C)/C_div[i])

                x = self.__conv_layer_3d__(C_over_div, 1, 'd_decompress_block_{}a'.format(i))(x)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

                x = self.__deconv_layer_3d__(C_over_div, 2, 'd_decompress_block_{}b'.format(i))(x)     
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

            skip_connection = x

            C = 4
            for i in range(12):
                B_skip_connection = x

                x = self.__conv_layer_3d__(C, 1, 'd_residual_block_{}a'.format(i))(x)
                x = tf.keras.layers.ReLU()(x)

                x = self.__conv_layer_3d__(C, 1, 'd_residual_block_{}b'.format(i))(x)

                x = tf.keras.layers.Add()([x, B_skip_connection])

            
            x = self.__conv_layer_3d__(C, 1, 'd_conv_2')(x)
            x = tf.keras.layers.Add()([x, skip_connection])
                
            x = self.__conv_layer_3d__(self.input.shape[-1], 1, 'd_conv_out')(x)

        return x

    def get_model(self):
        return tf.keras.Model(inputs=self.input, outputs=self.y)


if __name__ == "__main__":
    print("[+]: Converting AEflow model TF1 to Keras-compatible")
    utils.convert_aeflow_to_keras("models/AEflow1D_model/AEflow")