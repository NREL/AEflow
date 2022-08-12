from gc import callbacks
import math
import utils
import argparse
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

from matplotlib import cm
from AEflow3D import AEflow3D
from AEflow1D import AEflow1D

matplotlib.use('Agg')


class PreformanceCallback(tf.keras.callbacks.Callback):
    """
        PerformanceCallback: 
            This callback generates a reconstruction sample
            from the model being trained every 2 epochs
    """
    def __init__(self, exp_name, data, weights, mu_sig, f_writer):
        super().__init__()
        self.exp_name = exp_name
        self.data = data
        self.loss_w = weights
        self.f_writer = f_writer
        self.mu_sig = mu_sig

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % 2 == 0:
            mu_sig = self.mu_sig
            print("Reconstructing on_epoch_begin")
            y_true = self.data
            y_pred = self.model(y_true)

            y_std = mu_sig[1]*y_pred + mu_sig[0]
            tf.summary.image("full_rec", y_std[:1, :, :, 0, :1], step=epoch)
            utils.plot_data(y_std[:1], f'figs/{self.exp_name}/', f'{epoch}_epoch_begin')

            y_std = mu_sig[1]*y_true + mu_sig[0]
            utils.plot_data(y_std[:1], f'figs/{exp_name}/', 'original')
            print(f"\n[+]: Done generating images on epoch {epoch}\n")

def scheduler(epoch, lr):
    """
        Scheduler function callback
    """
    if epoch < 30:
        return lr
    elif epoch % 10 == 0:
        return lr * tf.math.exp(-0.1)   
    return lr

def loss_div(_, y_pred):
    """
        Divergence-free condition loss
    """
    h=2.*math.pi/128.
    bs = tf.cast(tf.shape(y_pred)[0], y_pred.dtype)
    elems = tf.cast(np.prod(y_pred.shape[1:]), y_pred.dtype)

    dv1_dx = y_pred[:, 1:, :, :, 0] - y_pred[:, :-1, :, :, 0]
    dv2_dy = y_pred[:, :, 1:, :, 1] - y_pred[:, :, :-1, :, 1]
    dv3_dz = y_pred[:, :, :, 1:, 2] - y_pred[:, :, :, :-1, 2]

    dv1_dx = tf.pad(dv1_dx, [(0,0),(0,1),(0,0),(0,0)])
    dv2_dy = tf.pad(dv2_dy, [(0,0),(0,0),(0,1),(0,0)])
    dv3_dz = tf.pad(dv3_dz, [(0,0),(0,0),(0,0),(0,1)])

    div = (dv1_dx + dv2_dy + dv3_dz) * (1 / h)

    return tf.reduce_sum(tf.square((div))) * (1/elems) * (1/bs)

def loss_enstr(y_true, y_pred):
    """
        Enstrophy preserving loss
    """
    bs = tf.cast(tf.shape(y_true)[0], y_true.dtype)
    elems = tf.cast(np.prod(y_pred.shape[1:]), y_pred.dtype)

    def loss_curl(data):
        h=2.*math.pi/128.

        dv1_dy = data[:, :, 1:, :, 0] - data[:, :, :-1, :, 0]
        dv2_dx = data[:, 1:, :, :, 1] - data[:, :-1, :, :, 1]
        dv3_dy = data[:, :, 1:, :, 2] - data[:, :, :-1, :, 2]
        dv1_dz = data[:, :, :, 1:, 0] - data[:, :, :, :-1, 0]
        dv2_dz = data[:, :, :, 1:, 1] - data[:, :, :, :-1, 1]
        dv3_dx = data[:, 1:, :, :, 2] - data[:, :-1, :, :, 2]

        dv1_dy = tf.pad(dv1_dy, [(0,0),(0,0),(0,1),(0,0)])
        dv2_dx = tf.pad(dv2_dx, [(0,0),(0,1),(0,0),(0,0)])
        dv3_dy = tf.pad(dv3_dy, [(0,0),(0,0),(0,1),(0,0)])
        dv1_dz = tf.pad(dv1_dz, [(0,0),(0,0),(0,0),(0,1)])
        dv2_dz = tf.pad(dv2_dz, [(0,0),(0,0),(0,0),(0,1)])
        dv3_dx = tf.pad(dv3_dx, [(0,0),(0,1),(0,0),(0,0)])

        curl = ((dv3_dy-dv2_dz) + (dv1_dz-dv3_dx) + (dv2_dx-dv1_dy)) * (1 / h)
        return tf.expand_dims(tf.reduce_sum(tf.square(curl)), 0) * (1/elems)

    return tf.reduce_sum(tf.square(((loss_curl(y_true) - loss_curl(y_pred))))) * (1/elems) * (1/bs)

def loss_mse(y_true, y_pred):
    """
        Mean Squared Error loss implemented for multiGPU utilization
        https://www.tensorflow.org/tutorials/distribute/custom_training
    """
    mse = tf.keras.losses.MeanSquaredError(tf.keras.losses.Reduction.NONE)
    per_example_loss = mse(y_true[..., tf.newaxis], y_pred[..., tf.newaxis]) / tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=_BS_)

def custom_loss(weights):
    """
        Main loss containing the three terms MSE, Divergence-free and Enstrophy
    """
    def _loss(y_true, y_pred):

        mean_squared = loss_mse(y_true, y_pred)
        divergence = loss_div(y_true, y_pred)
        enstrophy = loss_enstr(y_true, y_pred)

        loss = mean_squared*weights[0] + divergence*weights[1] + enstrophy*weights[2]

        return loss
    return _loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AEflow training')
    parser.add_argument('--model', type=str, help='Model to use', default="AEflow3D")
    parser.add_argument('--lr', type=str, help='learning rate', default="1e-4")
    parser.add_argument('--tr_epochs', type=int, help="Training epochs", default=200)
    parser.add_argument('--loss', type=str, help="Loss type mse/custom", default="custom")
    parser.add_argument('--bs', type=int, help="Batch size, max 8", default=4)
    parser.add_argument('--data_file', type=str, help="Data file", default="")
    parser.add_argument('--data_size', type=int, help="Amount of datapoints", default=1324)
    parser.add_argument('--losses_weights', type=str, help="Weights to assign to each loss: [MSE, Div-free, Enstrophy]", nargs=3, default=['1', '0', '0'])
    parser.add_argument("--test", type=bool, help="Whether this is a test run (adds 't_' as prefix to the name)", default=False)
    parser.add_argument('--prefix', type=str, help="Prefix name for the run", default="")
    args = parser.parse_args()

    test_data_3d = '' # Set test data for reconstruction callback
    nowtime = datetime.datetime.now().strftime("%m%d-%H%M")

    # -------- Setting of the variables -------- #
    global _BS_
    _BS_ = args.bs
    CHUNK_SZ = 1 # Chunk size to train with smaller pieces of the flow
    h = 2*math.pi/128
    mu_sig = [0, 1.28566] # mean and std of the data

    data_size = args.data_size
    loss = args.loss
    tr_lr = args.lr
    train_epochs = args.tr_epochs
    data_file = args.data_file
    spe =  int(data_size/_BS_)
    w_str = args.losses_weights

    m_name = "AE3D_" if args.model == "AEflow3D" else "AE1D_"
    test = "t_" if args.test else ""
    exp_name = f"{test}{nowtime}_{args.prefix}_{m_name}e{train_epochs}d{data_size}_{tr_lr}_bs{_BS_}_{w_str[0]}_{w_str[1]}_{w_str[2]}"
    models_file = f"models/{exp_name}/AEflow"
    loss_weights = list(map(float, w_str))
    
    n_channels = 3 if args.model == "AEflow3D" else 1
    # ------------------------------------------- #

    # Metrics and loss to use for training
    metrics = [loss_mse, loss_div, loss_enstr] if args.model == "AEflow3D" else loss_mse
    loss = custom_loss(loss_weights) if args.model == "AEflow3D" else loss_mse

    log_dir = "logs/fit/" + exp_name
    file_writer = tf.summary.create_file_writer(log_dir + '/train')
    file_writer.set_as_default()

    # -------- Print experiment details -------- #
    print(f"\nExperiment name: {exp_name}\n------\n\n"
        + f"Model: {models_file} | Data: {data_file}\n"
        + f"LR {args.lr} | Epochs {args.tr_epochs} ({spe} steps) | Loss {args.loss} | BS {args.bs * CHUNK_SZ**3} \n")
    # ------------------------------------------ #

    test_data = iter(utils.get_data_iter_from_tfrecord(test_data_3d, 1, mu_sig=mu_sig, bsize=1, n_channels=n_channels, bshuffle=1, chunk=CHUNK_SZ)).next()[0] 

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=models_file, save_weights_only=False, period=1, save_best_only=True, monitor="loss", mode="min"),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=12, mode='min'),
        tf.keras.callbacks.LearningRateScheduler(scheduler),
        PreformanceCallback(exp_name, test_data, loss_weights, mu_sig, file_writer),
    ]

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}\n'.format(strategy.num_replicas_in_sync))

    train_dataset = utils.get_data_iter_from_tfrecord(data_file, spe*train_epochs+1, mu_sig=mu_sig, n_channels=n_channels, bsize=_BS_, chunk=CHUNK_SZ) 

    with strategy.scope():
        x = tf.keras.Input(shape=[128//CHUNK_SZ, 128//CHUNK_SZ, 128//CHUNK_SZ, n_channels], dtype=tf.double)
        model = AEflow3D(x, "full") if "AEflow3D" == args.model else AEflow1D(x, "full")
        model = model.get_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(float(tr_lr)), loss=loss, metrics=metrics)

    model.fit(train_dataset, epochs=train_epochs, steps_per_epoch=spe, verbose=2, callbacks=callbacks)
    print(f"Model saved in {models_file}")