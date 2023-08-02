import os
import pickle
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint
from sklearn.neighbors import KernelDensity
import warnings

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D,Conv1D
from tensorflow.keras.layers import Conv2DTranspose,Conv1DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
from keras.layers import LeakyReLU

class AEmodels:
   
    def __init__(self,x_train):
        self.x_train = x_train    
    
    def printattr(self):
        print(self.x_train)
        
        
    def AEmodel1(self):
        model = keras.Sequential(
            [
                layers.Input(shape=(self.x_train.shape[1], self.x_train.shape[2])),
                layers.Conv1D(
                    filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
                ),

                layers.Conv1D(
                    filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                layers.Conv1DTranspose(
                    filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
                ),

                layers.Conv1DTranspose(
                    filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
            ]
        )
#         model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        model.summary()
        return(model)

    def ae_en_model1(self):
        warnings.filterwarnings("ignore")

        input_img = keras.Input(shape=(self.x_train.shape[1], self.x_train.shape[2]))

        x = Conv1D(filters=32, kernel_size=7, padding="same", strides=2, activation="relu")(input_img)
        encoded = Conv1D(filters=16, kernel_size=7, padding="same", strides=2, activation="relu")(x)
    
        x = Conv1DTranspose(filters=16, kernel_size=7, padding="same", strides=2, activation="relu")(encoded)
        x = Conv1DTranspose(filters=32, kernel_size=7, padding="same", strides=2, activation="relu")(x)

        decoded = Conv1DTranspose(filters=1, kernel_size=7, padding="same")(x)
        
      
        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)

#         autoencoder.summary()
        return(autoencoder,encoder)
    
    # conv2D - maxpooling - upsampling
    def AEmodel_enc1(self):
        warnings.filterwarnings("ignore")

        input_img = keras.Input(shape=(self.x_train.shape[1], self.x_train.shape[2], 1))

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',strides=2)(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same',strides=1)(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',strides=1)(x)
        # x = layers.MaxPooling2D((2, 2), padding='same',strides=1)(x)
        # x = layers.Conv2D(16, (3, 3), activation='relu', padding='same',strides=1)(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same',strides=1)(x)


        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',strides=1)(encoded)
        # x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',strides=1)(x)
        x = layers.UpSampling2D((2, 2))(x)
        # x = layers.Conv2D(64, (3, 3), activation='relu',strides=1)(x)
        # x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same',strides=1)(x)

        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)

        autoencoder.summary()
        return(autoencoder,encoder)


    def AEmodel2(self):
        warnings.filterwarnings("ignore")

        model = keras.Sequential(
            [
                # encoder
                layers.Input(shape=(self.x_train.shape[1], self.x_train.shape[2])),
                layers.Conv1D(
                    filters=32, kernel_size=9, padding="same", strides=2, activation="relu"
                ),
                layers.Dropout(rate=0.2),
                layers.MaxPooling1D(
                    pool_size=4, strides=1, padding="same"
                ),

                layers.Conv1D(
                    filters=16, kernel_size=9, padding="same", strides=2, activation="relu"
                ),
                 layers.Dropout(rate=0.2),
                layers.MaxPooling1D(
                    pool_size=4, strides=1, padding="same"
                ),
                layers.Conv1D(
                    filters=8, kernel_size=9, padding="same", strides=1, activation="relu"
                ),
                layers.Dropout(rate=0.2),
                layers.MaxPooling1D(
                    pool_size=4, strides=1, padding="same"
                ),
                # decoder
                layers.Conv1DTranspose(
                    filters=8, kernel_size=9, padding="same", strides=1, activation="relu"
                ),
                layers.Conv1DTranspose(
                    filters=16, kernel_size=9, padding="same", strides=2, activation="relu"
                ),
                layers.Dropout(rate=0.2),

                layers.Conv1DTranspose(
                    filters=32, kernel_size=9, padding="same", strides=2, activation="relu"
                ),
                layers.Conv1DTranspose(filters=1, kernel_size=9, padding="same"),
            ]
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mse")

        model.summary()
        return(model)

    def AEmodel3(self):

        warnings.filterwarnings("ignore")

        model = keras.Sequential(
            [
                # encoder
                layers.Input(shape=(self.x_train.shape[1], self.x_train.shape[2])),
                layers.Conv1D(
                    filters=32, kernel_size=9, padding="same", strides=2, activation="relu"
                ),
                layers.Dropout(rate=0.1),
                layers.MaxPooling1D(
                    pool_size=4, strides=1, padding="same"
                ),

                layers.Conv1D(
                    filters=16, kernel_size=9, padding="same", strides=2, activation="relu"
                ),
    #              layers.Dropout(rate=0.1),
                layers.MaxPooling1D(
                    pool_size=4, strides=1, padding="same"
                ),
                layers.Conv1D(
                    filters=8, kernel_size=9, padding="same", strides=1, activation="relu"
                ),
    # #             layers.Dropout(rate=0.1),
                layers.MaxPooling1D(
                    pool_size=4, strides=1, padding="same"
                ),
    #             # decoder
                layers.Conv1DTranspose(
                    filters=8, kernel_size=9, padding="same", strides=1, activation="relu"
                ),
    #             layers.Dropout(rate=0.2),

                layers.Conv1DTranspose(
                    filters=16, kernel_size=9, padding="same", strides=2, activation="relu"
                ),
    #             layers.Dropout(rate=0.2),

                layers.Conv1DTranspose(
                    filters=32, kernel_size=9, padding="same", strides=2, activation="relu"
                ),
    #             layers.Dropout(rate=0.2),

                layers.Conv1DTranspose(filters=1, kernel_size=9, padding="same"),
    #             layers.Dropout(rate=0.2),

            ]
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

        model.summary()
        return(model)
    
    

    
# class vaeModels:
#      def __init__(self,x_train):
#         self.x_train = x_train    
    
#     def printattr(self):
#         print(self.x_train)
        
        
#     def vaemodel1(self):




class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    
#     def loss_function(recon_x, x, mean, log_var):
#     RECON = F.mse_loss(recon_x, x.view(-1, 784), reduction='sum')
#     KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
#     return RECON + KLD, RECON, KLD

    def call(self, inputs):
        """Call the model on a particular input."""
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
#                     keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                      keras.losses.binary_crossentropy(data, reconstruction), axis=(1)
#                       keras.losses.MeanSquaredError(data, reconstruction), axis=(1)

                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    
#     def test_step(self, data):
#         """Step run during validation."""
#         if isinstance(data, tuple):
#             data = data[0]

#         z_mean, z_log_var, reconstruction = self(data)
#         reconstruction_loss = tf.reduce_mean(
#             tf.keras.losses.binary_crossentropy(data, reconstruction))
#         reconstruction_loss *= 100
#         kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
#         kl_loss = tf.reduce_mean(kl_loss)
#         kl_loss *= -0.5
#         total_loss = reconstruction_loss + kl_loss

#         return {
#             "loss": total_loss,
#             "reconstruction_loss": reconstruction_loss,
#             "kl_loss": kl_loss,
#         }


    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1)))
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        return {"loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss}
    
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder_decoder(latent_dim=2,seq_length=300):

    # encoder_inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
    encoder_inputs = keras.Input(shape=(seq_length, 1))

    x = layers.Conv1D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv1D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    z = Sampling()([z_mean, z_log_var])
    # z = layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
#     encoder.summary()

    decoder_input = keras.Input(shape=(latent_dim,))
    x = layers.Dense(75  * 64, activation="relu")(decoder_input)
    x = layers.Reshape((75, 64))(x)
    x = layers.Conv1DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv1DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv1DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(decoder_input, decoder_outputs, name="decoder")

#     decoder.summary()
    z_decoded = decoder(z)
    return encoder,decoder,z_decoded,encoder_inputs,decoder_outputs,z_mean,z_log_var

def build_encoder_decoder_TD(latent_dim=2,seq_length=300):

    # encoder_inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
    encoder_inputs = keras.Input(shape=(seq_length, 1))

    x = layers.Conv1D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv1D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    z = Sampling()([z_mean, z_log_var])
    # z = layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
#     encoder.summary()

    decoder_input = keras.Input(shape=(latent_dim,))
    x = layers.Dense(75  * 32, activation="relu")(decoder_input)
    x = layers.Reshape((75, 32))(x)
    x = layers.Conv1DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv1DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv1DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(decoder_input, decoder_outputs, name="decoder")
    
#     decoder.summary()
    z_decoded = decoder(z)
    return encoder,decoder,z_decoded,encoder_inputs,decoder_outputs,z_mean,z_log_var

def build_encoder_decoder_TD_LR(latent_dim=2,seq_length=300):

    # encoder_inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
    encoder_inputs = keras.Input(shape=(seq_length, 1))

    x = layers.Conv1D(64, 3, strides=2, padding="same")(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = layers.Conv1D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16)(x)
    x = LeakyReLU()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    z = Sampling()([z_mean, z_log_var])
    # z = layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
#     encoder.summary()

    decoder_input = keras.Input(shape=(latent_dim,))
    x = layers.Dense(75  * 32)(decoder_input)
    x = LeakyReLU(alpha=0.3)(x)
    x = layers.Reshape((75, 32))(x)
    x = layers.Conv1DTranspose(32, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = layers.Conv1DTranspose(64, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.3)(x)
    decoder_outputs = layers.Conv1DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(decoder_input, decoder_outputs, name="decoder")
    
#     decoder.summary()
    z_decoded = decoder(z)
    return encoder,decoder,z_decoded,encoder_inputs,decoder_outputs,z_mean,z_log_var


def build_encoder_decoder_TD_LR_lessN(latent_dim=2,seq_length=300):

    # encoder_inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
    encoder_inputs = keras.Input(shape=(seq_length, 1))

    x = layers.Conv1D(32, 3, strides=2, padding="same")(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = layers.Conv1D(16, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(8)(x)
    x = LeakyReLU()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    z = Sampling()([z_mean, z_log_var])
    # z = layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
#     encoder.summary()

    decoder_input = keras.Input(shape=(latent_dim,))
    x = layers.Dense(75  * 16)(decoder_input)
    x = LeakyReLU(alpha=0.3)(x)
    x = layers.Reshape((75, 16))(x)
    x = layers.Conv1DTranspose(16, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = layers.Conv1DTranspose(32, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.3)(x)
    decoder_outputs = layers.Conv1DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(decoder_input, decoder_outputs, name="decoder")
    
#     decoder.summary()
    z_decoded = decoder(z)
    return encoder,decoder,z_decoded,encoder_inputs,decoder_outputs,z_mean,z_log_var


    encoder_inputs = keras.Input(shape=(seq_length, 1))

    x = layers.Conv1D(16, 3, strides=2, padding="same")(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(8)(x)
    x = LeakyReLU()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
#     encoder.summary()


    decoder_input = keras.Input(shape=(latent_dim,))
    x = layers.Dense(75  * 16)(decoder_input)
    x = LeakyReLU(alpha=0.3)(x)
    x = layers.Reshape((75, 16))(x)
    x = layers.Conv1DTranspose(16, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = layers.Conv1DTranspose(8, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.3)(x)
    decoder_outputs = layers.Conv1DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(decoder_input, decoder_outputs, name="decoder")

#     decoder.summary()
    z_decoded = decoder(z)
    return encoder,decoder,z_decoded,encoder_inputs,decoder_outputs,z_mean,z_log_var


def build_encoder_decoder_TD_mp(latent_dim=2,seq_length=300):

    # encoder_inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
    encoder_inputs = keras.Input(shape=(seq_length, 1))

    x = layers.Conv1D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = Conv1D(64, 3, activation = 'relu', padding = "SAME")(x)
    x = layers.Conv1D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = Conv1D(32, 3, activation = 'relu', padding = "SAME")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
#     x = layers.Dropout(0.5)(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    z = Sampling()([z_mean, z_log_var])
    # z = layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
#     encoder.summary()

    decoder_input = keras.Input(shape=(latent_dim,))
    x = layers.Dense(75  * 32, activation="relu")(decoder_input)
    x = layers.Reshape((75, 32))(x)
    x = layers.Conv1DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv1DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv1DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(decoder_input, decoder_outputs, name="decoder")
    
#     decoder.summary()
    z_decoded = decoder(z)
    return encoder,decoder,z_decoded,encoder_inputs,decoder_outputs,z_mean,z_log_var

class CustVariationalLayer(keras.layers.Layer):
    
    def vae_loss(self, x, z, mu,log_var):
        # The references to the layers are resolved outside the function 
        x = K.flatten(x)   
        z = K.flatten(z)
        
        # reconstruction loss per sample 
        # Note: that this is averaged over all features (e.g.. 784 for MNIST) 
        reco_loss = tf.keras.metrics.binary_crossentropy(x, z)
        
        # KL loss per sample - we reduce it by a factor of 1.e-3 
        # to make it comparable to the reco_loss  
        kln_loss  = -0.5e-4 * K.mean(1 + log_var - K.square(mu) - K.exp(log_var), axis=1) 
        # mean per batch (axis = 0 is automatically assumed) 
        return K.mean(reco_loss + kln_loss), K.mean(reco_loss), K.mean(kln_loss) 
           
    def call(self, inputs):
        inp_img = inputs[0]
        out_img = inputs[1]
        mu = inputs[2]
        sigma = inputs[3]
        total_loss, reco_loss, kln_loss = self.vae_loss(inp_img, out_img,mu,sigma)
        self.add_loss(total_loss, inputs=inputs)
        self.add_metric(total_loss, name='total_loss', aggregation='mean')
        self.add_metric(reco_loss, name='reco_loss', aggregation='mean')
        self.add_metric(kln_loss, name='kl_loss', aggregation='mean')
        
        return out_img  #not really used in this approach  

class calc_output_with_los(keras.layers.Layer):

    def vae_loss(self, x, z_decoded,mu,sigma):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)

        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)

        kl_loss = -5e-4 * K.mean(1 + sigma - K.square(mu) - K.exp(sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        mu = inputs[2]
        sigma = inputs[3]
        loss = self.vae_loss(x, z_decoded,mu,sigma)
        self.add_loss(loss, inputs=inputs)
        return x

def build_vae3(encoder,decoder,z_decoded,inputs,mu,sigma):
    outputs = CustVariationalLayer()([inputs, z_decoded,mu,sigma])
    vae = keras.Model(inputs, outputs)

    vae.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss=None)
    return(vae)


def build_vae2(encoder,decoder,z_decoded,inputs,mu,sigma):
    outputs = calc_output_with_los()([inputs, z_decoded,mu,sigma])
    vae = keras.Model(inputs, outputs)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=None)
    return(vae)



def build_vae(encoder,decoder):
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))#loss=vae_loss
    return(vae)