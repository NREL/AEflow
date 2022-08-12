import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Path for data to load
orig_data = np.load('data/3D_datapoint.npy')
embedding_layer = 'e_conv_out'

full_model = tf.keras.models.load_model("models/AEflow3D_model/AEflow", compile=False)
compress_model = tf.keras.models.Model([full_model.inputs], [full_model.get_layer(embedding_layer).output])

# Runs the full AE (i.e., both the encoder and decoder)
reconstr_data = full_model.predict(orig_data)
embed_data = compress_model.predict(orig_data)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(orig_data[0, 0, :, :, 0])
ax[1].imshow(reconstr_data[0, 0, :, :, 0])
ax[2].imshow(embed_data[0, 0, :, :, 0])
plt.show()
