import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def _bytes_feature_(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature_(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _generate_tfrecords_(x, record_path):
    with tf.python_io.TFRecordWriter(record_path) as writer:
        N, d0, d1, d2, c  = x.shape
            
        for i in range(N):
            example = tf.train.Example(features=tf.train.Features(feature={
                                         'index': _int64_feature_(i),
                                          'data': _bytes_feature_(x[i, ...].tostring()),
                                            'd0': _int64_feature_(d0),
                                            'd1': _int64_feature_(d1),
                                            'd2': _int64_feature_(d2),
                                             'c': _int64_feature_(c)}))
            writer.write(example.SerializeToString())
        
# Parser function for data pipeline
def _parse_(serialized_example, mu_sig=None):
    feature = {'index': tf.FixedLenFeature([], tf.int64),
                'data': tf.FixedLenFeature([], tf.string),
                  'd0': tf.FixedLenFeature([], tf.int64),
                  'd1': tf.FixedLenFeature([], tf.int64),
                  'd2': tf.FixedLenFeature([], tf.int64),
                   'c': tf.FixedLenFeature([], tf.int64)}
    example = tf.parse_single_example(serialized_example, feature)
    
    idx = example['index']
    d0, d1, d2, c = example['d0'], example['d1'], example['d2'], example['c']
    data = tf.decode_raw(example['data'], tf.float64)
    data = tf.reshape(data, (d0, d1, d2, c))

    if mu_sig is not None:
        data = (data - mu_sig[0])/mu_sig[1]
        
    return idx, data

def plot_data(data_path, img_path, s=64):
    data = np.load(data_path)
    if data.ndim == 5:
        data = data[..., 0]

    if not os.path.exists(img_path):
        os.makedirs(img_path)

    for i in range(data.shape[0]):
      plt.figure(figsize=(6, 6))
      plt.imshow(data[i, :, :, s], cmap='viridis')
      plt.xticks([])
      plt.yticks([])
      plt.savefig(img_path+'img{0:04d}.png'.format(i), dpi=200, bbox_inches='tight')
      plt.close()