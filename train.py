import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import tensorflow as tf
from AEflow import AEflow
import utils


def train(data_path, mu_sig, batch_size=1, N_epochs=10):
    print('Initializing network ...', end=' ')
    x = tf.placeholder(tf.float64, [None, 128, 128, 128, 1])
    model = AEflow(x, status='full')
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
    train_op = optimizer.minimize(model.loss, var_list=model.var_list)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=10000, var_list=model.var_list)
    print('Done.')

    print('Building data pipeline ...', end=' ')
    ds = tf.data.TFRecordDataset(data_path)
    ds = ds.map(lambda xx: utils._parse_(xx, mu_sig)).shuffle(1000).batch(batch_size)
    iterator = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)
    idx, x_out = iterator.get_next()
    init_iter_train  = iterator.make_initializer(ds)
    print('Done.')

    with tf.Session() as sess:
        print('Training network ...')
        sess.run(init)

        iters = 0
        for epoch in range(1, N_epochs+1):
            print('Epoch: '+str(epoch))
            
            # Loop through training data
            sess.run(init_iter_train)
            try:
                epoch_loss, N_train = 0, 0
                while True:
                    iters += 1

                    x_batch = sess.run(x_out)
                    sess.run(train_op, feed_dict={x: x_batch})
                    l = sess.run(model.loss, feed_dict={x: x_batch})

                    if (iters % print_every) == 0:
                        print('Iterations=%d, G loss=%.5f' %(iters, l))

            except tf.errors.OutOfRangeError:
                pass
            
        if not os.path.exists('new_model'):
            os.makedirs('new_model')
        saver.save(sess, 'new_model/autoencoder')

    print('Done.')


# Path for data to load
training_data = 'data/example_data.tfrecord'

# Mean and standard deviation to normalize data
mu_sig = [0, 1.26436]

# Example script for training the network
train(training_data, mu_sig=mu_sig)


