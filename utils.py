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
def _parse_(serialized_example, mu_sig=None, n_channels=3, chunk=None):
    feature = {'index': tf.io.FixedLenFeature([], tf.int64),
                'data': tf.io.FixedLenFeature([], tf.string),
                  'd0': tf.io.FixedLenFeature([], tf.int64),
                  'd1': tf.io.FixedLenFeature([], tf.int64),
                  'd2': tf.io.FixedLenFeature([], tf.int64),
                  'c': tf.io.FixedLenFeature([], tf.int64)}
    example = tf.io.parse_single_example(serialized_example, feature)

    idx = example['index']
    d0, d1, d2, c = example['d0'], example['d1'], example['d2'], example['c']

    data = tf.io.decode_raw(example['data'], tf.float64)
    data = tf.reshape(data, (d0, d1, d2, c))
    
    # Retrieve only the x channel if data is 1D
    if n_channels == 1:
        data = data[..., 0]

    if mu_sig is not None:
        data = (data - mu_sig[0])/mu_sig[1]

    # Reshaping of data when chunk size is enabled
    if chunk > 1:
        data = tf.reshape(data, (chunk, d0//chunk, chunk, d1//chunk, chunk, d2//chunk, c))
        data = tf.transpose(data, (0, 2, 4, 1, 3, 5, 6))
        data = tf.reshape(data, (-1, d0//chunk, d1//chunk, d2//chunk, c))

    return data, data

def plot_data(data, img_path, prefix="", s=0):
    if data.ndim == 5:
        data = data[..., 0]

    if not os.path.exists(img_path):
        os.makedirs(img_path)

    for i in range(data.shape[0]):
      plt.figure(figsize=(6, 6))
      plt.imshow(data[i, :, :, s], cmap='viridis')
      plt.xticks([])
      plt.yticks([])
      plt.savefig(img_path+'{}_img{}.png'.format(prefix, i), dpi=200, bbox_inches='tight')
      plt.close()

def model_summary():
    import tensorflow.contrib.slim as slim
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_data_iter_from_tfrecord(data_path='data/example_data.tfrecord', repeat=1, mu_sig=[0, 1.26436], n_channels=3, bsize=1, bshuffle=32, chunk=1):

    ds = tf.data.TFRecordDataset(data_path)
    ds = ds.map(lambda xx: _parse_(xx, mu_sig, n_channels=n_channels, chunk=chunk)).shuffle(bshuffle).repeat(repeat).batch(bsize)
    
    if chunk > 1:
        h, w, d, c = ds.take(1).get_single_element()[0].shape[-4:]
        new_shape = (-1, h, w, d, c)
        ds = ds.map(lambda x, y: (tf.reshape(x, new_shape), tf.reshape(y, new_shape)))

    return ds

def convert_aeflow_to_keras(path):
    def _mse(y_true, y_pred):
        return tf.reduce_mean((y_true - y_pred)**2)

    #Import both AEflow and the new AEflow1D model
    from AEflow import AEflow
    from AEflow1D import AEflow1D

    # Disable eager execution for AEflow Placeholder
    tf.compat.v1.disable_eager_execution()

    # Create and restore the original AEflow model
    x = tf.compat.v1.placeholder(tf.float64, [None, None, None, None, 1])
    aeflow_model = AEflow(x, status='full')
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver(var_list=aeflow_model.var_list)

    # Save the original AEflow weights
    with tf.compat.v1.Session() as sess:
        print('Loading saved network ...', end=' ')
        sess.run(init)
        saver.restore(sess, 'weights/model_aeflow_v1/AEflow')
        aeflow_weights = sess.run(aeflow_model.var_list)

    # Reset default graph and enable back eager execution to load model
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.enable_eager_execution()

    # Load model
    x = tf.keras.Input(shape=[None, None, None, 1], dtype=tf.float64)
    model = AEflow1D(x, status='full')
    model = tf.keras.Model(inputs=x, outputs=model.y)

    # Get only the same layers that the original model has
    layers_to_restore = []
    for l in model.layers:
        if 'e_' != l.name[:2] and 'd_' != l.name[:2]:
            continue
        layers_to_restore.append(l)

    # Input the AEflow weights into the AEflow1D layers and save the model
    for i, (l, w) in enumerate(zip(layers_to_restore, aeflow_weights)):
        print(f"Loading weights layer {l.name} number {i}")
        l.set_weights([w])

    model.save(path)
    print("Saved Keras model in ", path)
    
    mu_sig = [0, 1.26436]
    x_batch = iter(get_data_iter_from_tfrecord()).next()[1]

    y_out = model.predict(x_batch)
    y_out = mu_sig[1]*y_out + mu_sig[0]

    np.save("tmp/example_keras_data.npy", y_out[..., 0])

    print("Printing AEflow Keras data")
    plot_data('tmp/example_keras_data.npy', 'figs/')

    print("Evaluating model")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=_mse, metrics=['mse'])
    print(model.evaluate(x_batch, x_batch))

def show_example_reconstructed(model):
    mu_sig = [0, 1.26436]
    x_batch = get_data_iter_from_tfrecord().next()

    y_out = model.predict(x_batch)
    y_out = mu_sig[1]*y_out + mu_sig[0]

    np.save("tmp/custom_reconstruct.npy", y_out[..., 0])
    plot_data('tmp/custom_reconstruct.npy', 'figs/')
    print("Custom reconstruction done")

    print("Evaluating model")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='mse', metrics=['mse'])
    print(model.evaluate(x_batch, x_batch))