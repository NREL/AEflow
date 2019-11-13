import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import tensorflow as tf
from AEflow import AEflow
import utils

def run_full(data_path, mu_sig=None, batch_size=1):
    print('Initializing network ...', end=' ')
    tf.reset_default_graph()
    x = tf.placeholder(tf.float64, [None, None, None, None, 1])
    model = AEflow(x, status='full')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(var_list=model.var_list)
    print('Done.')

    print('Building data pipeline ...', end=' ')
    ds = tf.data.TFRecordDataset(data_path)
    ds = ds.map(lambda xx: utils._parse_(xx, mu_sig)).batch(batch_size)
    iterator = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)
    idx, x_out = iterator.get_next()
    init_iter_test  = iterator.make_initializer(ds)
    print('Done.')

    with tf.Session() as sess:
        print('Loading saved network ...', end=' ')
        sess.run(init)
        saver.restore(sess, 'model/AEflow')
        print('Done.')

        print('Running data ...')
        sess.run(init_iter_test)
        try:
            data_out = None
            while True:
                x_batch = sess.run(x_out)
                y_out = sess.run(model.y, feed_dict={x: x_batch})

                y_out = mu_sig[1]*y_out + mu_sig[0]
                if data_out is None:
                    data_out = y_out
                else:
                    data_out = np.concatenate((data_out, y_out), axis=0)

                print(data_out.shape)

        except tf.errors.OutOfRangeError:
            pass

    np.save('data/reconstructed_data.npy', data_out[..., 0])

    print('Done.')


def run_compress(data_path, mu_sig=None, batch_size=1):
    print('Initializing network ...', end=' ')
    tf.reset_default_graph()
    x = tf.placeholder(tf.float64, [None, None, None, None, 1])
    model = AEflow(x, status='compress')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(var_list=model.var_list)
    print('Done.')

    print('Building data pipeline ...', end=' ')
    ds = tf.data.TFRecordDataset(data_path)
    ds = ds.map(lambda xx: utils._parse_(xx, mu_sig)).batch(batch_size)
    iterator = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)
    idx, x_out = iterator.get_next()
    init_iter_test  = iterator.make_initializer(ds)
    print('Done.')

    with tf.Session() as sess:
        print('Loading saved network ...', end=' ')
        sess.run(init)
        saver.restore(sess, 'model/AEflow')
        print('Done.')

        print('Running data ...')
        sess.run(init_iter_test)
        try:
            data_out = None
            while True:
                x_batch = sess.run(x_out)
                y_out = sess.run(model.y, feed_dict={x: x_batch})

                if data_out is None:
                    data_out = y_out
                else:
                    data_out = np.concatenate((data_out, y_out), axis=0)

                print(data_out.shape)

        except tf.errors.OutOfRangeError:
            pass

    np.save('data/compressed_data.npy', data_out)
    utils._generate_tfrecords_(data_out, 'data/compressed_data.tfrecord')

    print('Done.')

def run_reconstruct(data_path, mu_sig=None, batch_size=1):
    print('Initializing network ...', end=' ')
    tf.reset_default_graph()
    x = tf.placeholder(tf.float64, [None, None, None, None, 8])
    model = AEflow(x, status='reconstruct')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(var_list=model.var_list)
    print('Done.')

    print('Building data pipeline ...', end=' ')
    ds = tf.data.TFRecordDataset(data_path)
    ds = ds.map(lambda xx: utils._parse_(xx, [0, 1])).batch(batch_size)
    iterator = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)
    idx, x_out = iterator.get_next()
    init_iter_test  = iterator.make_initializer(ds)
    print('Done.')

    with tf.Session() as sess:
        print('Loading saved network ...', end=' ')
        sess.run(init)
        saver.restore(sess, 'model/AEflow')
        print('Done.')

        print('Running data ...')
        sess.run(init_iter_test)
        try:
            data_out = None
            while True:
                x_batch = sess.run(x_out)
                y_out = sess.run(model.y, feed_dict={x: x_batch})

                y_out = mu_sig[1]*y_out + mu_sig[0]
                if data_out is None:
                    data_out = y_out
                else:
                    data_out = np.concatenate((data_out, y_out), axis=0)

        except tf.errors.OutOfRangeError:
            pass

    np.save('data/reconstructed_data2.npy', data_out[..., 0])

    print('Done.')


# Path for data to load
full_data = 'data/example_data.tfrecord'
compressed_data = 'data/compressed_data.tfrecord'

# Mean and standard deviation to normalize data
mu_sig = [0, 1.26436]

# Runs the full AE (i.e., both the encoder and decoder)
run_full(full_data, mu_sig=mu_sig)

# Runs only the encoder
run_compress(full_data, mu_sig=mu_sig)

# Runs only the encoder
run_reconstruct(compressed_data, mu_sig=mu_sig)

# Plot slices of the reconstructed/compressed data for reference
utils.plot_data('data/reconstructed_data.npy', 'figs/reconstructed_data/')
utils.plot_data('data/compressed_data.npy', 'figs/compressed_data/', s=8)
utils.plot_data('data/reconstructed_data2.npy', 'figs/reconstructed_data2/')
