import os
import utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

np.random.seed(42)

def do_embeddings_experiments(data, model, layer_name):
    model_compress = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output])

    if not os.path.exists('embed_data_3d.npy'):
        data = tf.data.Dataset.from_tensor_slices(np.load(data)).batch(5)
        embed = []
        for x_batch in data:
            pred = model_compress.predict(x_batch).reshape(x_batch.shape[0],-1)
            embed.append(pred)
        
        pred = np.vstack(embed)
        np.save('embed_data_3d', pred)
    else:
        pred = np.load('embed_data_3d')


    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, learning_rate='auto', init='random')
    tsvd = TruncatedSVD(n_components=2, random_state=42)

    pred_pca = pca.fit_transform(pred)
    pred_tsne = tsne.fit_transform(pred)
    pred_tsvd = tsvd.fit_transform(pred)

    fig, axes = plt.subplots(3)
    fig.set_figwidth(15)
    fig.set_figheight(5)
    plt.subplot(131).scatter(pred_pca[:, 0], pred_pca[:, 1])
    plt.subplot(131).set_title("PCA")
    plt.subplot(132).scatter(pred_tsne[:, 0], pred_tsne[:, 1], c='g')
    plt.subplot(132).set_title("t-SNE")
    plt.subplot(133).scatter(pred_tsvd[:, 0], pred_tsvd[:, 1], c='m')
    plt.subplot(133).set_title("Truncated SVD")

    print(f"PCA: Variance ratio {pca.explained_variance_ratio_} \n\t Singular Values {pca.singular_values_}")
    print(f"TSNE: KL divergence {tsne.kl_divergence_}")
    print(f"TSVD: Variance ratio {tsvd.explained_variance_ratio_} \n\t Singular Values {tsvd.singular_values_}")

    print("SAVED")

def do_saliency_experiments(model, data_file):
    from tf_keras_vis.saliency import Saliency
    def score_function(pred):
        loss = tf.keras.losses.MeanSquaredError()
        true = iter(utils.get_data_iter_from_tfrecord(data_file, bsize=1, bshuffle=1)).next()[0]

        return loss(true, pred)

    # Create Saliency object.
    saliency = Saliency(model, clone=True)

    # Generate saliency map
    X = iter(utils.get_data_iter_from_tfrecord(data_file, bsize=1, bshuffle=1)).next()[0]
    saliency_map = saliency(score_function, X, smooth_samples=20, smooth_noise=0.20)
    plt.imsave("figs/axis0.png", np.mean(saliency_map.squeeze(), axis=0))
    plt.imsave("figs/axis1.png", np.mean(saliency_map.squeeze(), axis=1))
    plt.imshow("figs/axis2.png", np.mean(saliency_map.squeeze(), axis=2))
    print("Done saliency")

def do_reconstruction(model, data_file, save_dir):
    mse = tf.keras.losses.MeanSquaredError()

    print("[+]: Performing reconstruction")
    mu_sig = [0, 1.28566]
    
    chunk_sz = int(128/model.input.shape[1])

    train_dataset = utils.get_data_iter_from_tfrecord(data_file, bsize=1, bshuffle=1, chunk=chunk_sz) 
    data = iter(train_dataset).next()[0]

    model_1d = tf.keras.models.load_model("models/AEflow1D_model/AEflow/")
    y_pred_1d = model_1d.predict(data[..., 0])
    y_pred_3d = model.predict(data)

    y_true = mu_sig[1]*data + mu_sig[0]
    utils.plot_data(y_true[:1], f'{save_dir}/', 'ground_truth')

    y_pred_3d = mu_sig[1]*y_pred_3d + mu_sig[0]
    utils.plot_data(y_pred_3d[:1], f'{save_dir}/', 'AEflow3D')  

    y_pred_1d = mu_sig[1]*y_pred_1d + mu_sig[0]
    utils.plot_data(y_pred_1d, f'{save_dir}/', 'AEflow1D')

    mse_1d = mse(y_true[..., 0], y_pred_1d[..., 0]).numpy()
    mse_3d = mse(y_true[..., 0], y_pred_3d[..., 0]).numpy()

    with open(f"{save_dir}/mse", "w+") as f:
        f.write(f"MSE 1D: {mse_1d}")
        f.write(f"MSE 3D: {mse_3d}")

    print(f"AEflow 1D MSE: {mse_1d}")
    print(f"AEflow 3D MSE: {mse_3d}")
    print(f"[+]: Reconstruction done in {save_dir}")

def do_kernel_features_experiments(model, exp_name, data_file, axis_cut=0):
    n_channels = model.input.shape[-1]
    img_tensor = iter(utils.get_data_iter_from_tfrecord(data_file, 1, mu_sig=[0, 1.2856], bsize=1, bshuffle=1, n_channels=n_channels)).next()[0]
    layer_names = ['e_conv_1', 'e_conv_2', 'e_conv_out', 'd_conv_2', 'd_decompress_block_2a']

    layer_outputs = [layer.output for layer in model.layers if layer.name in layer_names]
    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    intermediate_activations = activation_model.predict(img_tensor)

    images_per_row = 12
    max_images = 12
    # Now let's display our feature maps
    GRID = []
    for layer_name, layer_activation in zip(layer_names, intermediate_activations):
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]
        print(f"FEATURES For {layer_name}: {n_features}")
        n_features = min(n_features, max_images)

        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]

        # We will tile the activation channels in this matrix
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                layer_activation = np.squeeze(layer_activation)
                mid = layer_activation.shape[axis_cut]//2
                channel_image = layer_activation.take(mid, axis=axis_cut)[..., col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                            row * size : (row + 1) * size] = channel_image

        GRID.append(display_grid)
    display_grid = np.vstack(GRID)
    # Display the grid
    scale = 2. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.axis('off')
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.tight_layout()
    plt.savefig(f"figs/{exp_name}/{axis_cut}_{layer_name}")
        
    plt.show()

def do_gradcam_experiments_keras_vis(model_files, data_file, layer_name):
    from matplotlib import cm
    from tf_keras_vis.gradcam import Gradcam
    
    def model_modifier_function(current_model):
        target_layer = current_model.get_layer(name=layer_name)
        target_layer.activation = tf.keras.activations.linear
        new_model = tf.keras.Model(inputs=current_model.inputs, outputs=target_layer.output)
        return new_model

    def score(n_channels):
        def score_function(pred):
            loss = tf.keras.losses.MeanSquaredError()
            true = iter(utils.get_data_iter_from_tfrecord(data_file, bsize=1, bshuffle=1, n_channels=n_channels)).next()[0]
            print("In score ", true.shape)

            return loss(true, pred)
        return score_function

    axes_names = ["X", "Y", "Z"]
    fig = plt.figure(constrained_layout=True)
    fig.suptitle('Grad-CAM saliency')

    subfigs = fig.subfigures(nrows=len(os.listdir(model_files)), ncols=1)
    for k, name in enumerate(os.listdir(model_files)):
        model_file = f"{model_files}/{name}/AEflow"
        model = tf.keras.models.load_model(model_file, compile=False)
        n_channels = model.input.shape[-1]
        
        title = "AEflow 3D [λ:{} | β:{}]".format(name.split("_")[-2], name.split("_")[-1]) if n_channels == 3 else "AEflow 1D"

        subfigs[k].suptitle(title, fontsize=20)
        ax = subfigs[k].subplots(nrows=1, ncols=3)
    
        X = iter(utils.get_data_iter_from_tfrecord(data_file, bsize=1, bshuffle=1, n_channels=n_channels)).next()[0]
        X = tf.expand_dims(X, -1) if n_channels == 1 else X

        saliency = Gradcam(model, model_modifier=model_modifier_function, clone=False)
        cam = saliency(score(n_channels), X, penultimate_layer=-1)
        heatmap_ = (np.uint8(cm.jet(np.squeeze(cam)) * 255))[..., 2]
        
        X = np.squeeze(X)
        for ai, an in enumerate(axes_names):
            img = 42 

            heatmap = heatmap_.take(img, axis=ai) 
            X_ = X.take(img, axis=ai) if n_channels == 1 else X.take(img, axis=ai)[..., ai]

            # heatmap = np.mean(heatmap_, axis=ai)
            # X_ = np.mean(X, axis=ai) if n_channels == 1 else np.mean(X, axis=ai)[..., ai]

            ax[ai].set_title(an)
            ax[ai].imshow(X_) 
            ax[ai].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
            ax[ai].axis('off')
        
    plt.savefig("tmp/FULL")
    plt.show()

def do_dissipation_experiments(models_dir, data_file):
    import time
    import pprint

    def calculate_dissipation_divergence_curl(U):
        nu = 0.00542
        h=2.*np.pi/128.
        U = np.squeeze(U)
        
        dU0dx0, dU0dx1, dU0dx2 = np.gradient(U[..., 0], h, axis=(0,1,2))
        dU1dx0, dU1dx1, dU1dx2 = np.gradient(U[..., 1], h, axis=(0,1,2))
        dU2dx0, dU2dx1, dU2dx2 = np.gradient(U[..., 2], h, axis=(0,1,2))
        
        #fluctuating velocity strain rate
        s00 = dU0dx0
        s01 = 0.5*(dU0dx1+dU1dx0)
        s02 = 0.5*(dU0dx2+dU2dx0)
        s11 = dU1dx1
        s12 = 0.5*(dU1dx2+dU2dx1)
        s22 = dU2dx2

        e = 2*nu*(s00**2 + 2*s01**2 + 2*s02**2 + s11**2 + 2*s12**2 + s22**2)
        
        dis = np.mean(e)
        div = np.mean(dU0dx0+dU1dx1+dU2dx2)

        aux_curl = ((dU2dx1-dU1dx2) + (dU0dx2-dU2dx0) + (dU1dx0-dU0dx1)) #((dv3_dy-dv2_dz) + (dv1_dz-dv3_dx) + (dv2_dx-dv1_dy))
        curl = np.mean(np.square(aux_curl))
        return dis, div, curl

    mse = tf.keras.losses.MeanSquaredError()

    MAX = 1300
    mu_sig = [0, 1.28566]

    model_files = os.listdir(models_dir)

    true_dis, true_div, true_curl = 0, 0, 0
    dataset_iter = iter(utils.get_data_iter_from_tfrecord(data_file, 1, mu_sig=mu_sig, bshuffle=1, bsize=1))
    for i, (y_true, _) in enumerate(dataset_iter):
        diss, div, curl = calculate_dissipation_divergence_curl(y_true)
        true_dis += diss
        true_div += div
        true_curl += curl
        if i == MAX: break

    true_dis /= (i+1)
    true_div /= (i+1)
    true_curl /= (i+1)

    print(f"True dissipation: {true_dis}")
    print(f"True div: {true_div}")
    print(f"True curl: {true_curl}")

    start = time.time()
    models_dict = {"true": {"diss": true_dis, "div": true_div, "curl": true_curl}}
    for model_file in model_files:
        print("[+]: Model file being processed: ", model_file)
        dataset_iter = iter(utils.get_data_iter_from_tfrecord(data_file, 1, mu_sig=mu_sig, bshuffle=1, bsize=1))
        
        model = tf.keras.models.load_model(f"{models_dir}/{model_file}/AEflow", compile=False)
        models_dict[model_file] = {"diss": 0, "mse": 0, "div": 0, "curl": 0}
        i_start = time.time()
        for i, (y_true, _) in enumerate(dataset_iter):
            y_pred = model.predict(y_true, verbose=0)

            diss, div, curl = calculate_dissipation_divergence_curl(y_pred)
            models_dict[model_file]["diss"] += diss
            models_dict[model_file]["div"] += div
            models_dict[model_file]["mse"] += mse(y_true, y_pred).numpy()
            models_dict[model_file]["curl"] += curl

            if i == MAX: break

        models_dict[model_file]['diss'] /= (i+1)
        models_dict[model_file]['div'] /= (i+1)
        models_dict[model_file]['mse'] /= (i+1)
        models_dict[model_file]['curl'] /= (i+1)

        print(f"AVGs for {model_file}:")
        print(f"\t Divergence: {models_dict[model_file]['div']}")
        print(f"\t Dissipation rate {models_dict[model_file]['diss']}")
        print(f"\t MSE {models_dict[model_file]['mse']}")
        print(f"\t Curl {models_dict[model_file]['curl']}")

        print("[+]: Time taken for 1 model: {:.2f}m".format((time.time()-i_start)/60))

    pprint.pprint(models_dict)
    print("Total time taken {:.2f}m".format((time.time()-start)/60))
    print()

if __name__ == "__main__":
    test_data_3d = "" # 3-channel test data file in tfrecord format
    exp_name = "" # Folder to the specific model to evaluate

    # Load the model and its weights
    model_file = f"models/{exp_name}/AEflow"
    model = tf.keras.models.load_model(model_file, compile=False)
    
    selected_models = "models/evaluate/" # Folder to the models being evaluated
    do_reconstruction(model, test_data_3d, save_dir="tmp/")
    do_kernel_features_experiments(model, exp_name, test_data_3d)
    do_gradcam_experiments_keras_vis(selected_models, test_data_3d, layer_name='d_conv_out')
    do_dissipation_experiments(selected_models, test_data_3d)