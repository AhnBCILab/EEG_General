import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import scipy.io
from scipy.signal import stft, istft
import random
import pickle
import gzip
import tensorflow as tf
from tensorflow.keras import (
    layers,
    models,
    optimizers,
    utils,
    callbacks,
    metrics,
    losses,
    activations,
)

snum = 1
LEFT = 0
RIGHT = 1
C3 = 12
C4 = 49 #18

# info
eeg_ch_names = [
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
    "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
    "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
    "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
    "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
]
emg_ch_names = ["EMG1", "EMG2", "EMG3", "EMG4"]
ch_names = eeg_ch_names + emg_ch_names + ["Stim"]
ch_types = ["eeg"] * 64 + ["emg"] * 4 + ["stim"]
srate = 512

import math

def linear_diffusion_schedule(diffusion_times):
    min_rate = 0.0001
    max_rate = 0.02
    betas = min_rate + diffusion_times * (max_rate - min_rate)
    alphas = 1 - betas
    alpha_bars = tf.math.cumprod(alphas)
    signal_rates = tf.sqrt(alpha_bars)
    noise_rates = tf.sqrt(1 - alpha_bars)
    return noise_rates, signal_rates

def cosine_diffusion_schedule(diffusion_times):
    signal_rates = tf.cos(diffusion_times * math.pi / 2)
    noise_rates = tf.sin(diffusion_times * math.pi / 2)
    return noise_rates, signal_rates

def offset_cosine_diffusion_schedule(diffusion_times):
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    start_angle = tf.acos(max_signal_rate)
    end_angle = tf.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)

    return noise_rates, signal_rates

T = 1000
diffusion_times = tf.convert_to_tensor([x / T for x in range(T)])
linear_noise_rates, linear_signal_rates = linear_diffusion_schedule(
    diffusion_times
)
cosine_noise_rates, cosine_signal_rates = cosine_diffusion_schedule(
    diffusion_times
)
(
    offset_cosine_noise_rates,
    offset_cosine_signal_rates,
) = offset_cosine_diffusion_schedule(diffusion_times)

# modeling
import keras
#@keras.saving.register_keras_serializable()
def sinusoidal_embedding(x):
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(1.0),
            tf.math.log(1000.0),
            32 // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings

gpus = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {gpus}")
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    print(f"Using MirroredStrategy with {len(gpus)} GPUs")
else:
    strategy = tf.distribute.get_strategy()  # Use default strategy (CPU or single GPU)
    print("Using default strategy (CPU or single GPU)")

def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply

def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply

# U-Net
IMAGE_SIZE = 64
NUM_CLASSES = 2

with strategy.scope():
    noisy_images = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = layers.Conv2D(32, kernel_size=1)(noisy_images)
    
    noise_variances = layers.Input(shape=(1, 1, 1))
    class_labels = layers.Input(shape=(NUM_CLASSES,))  # One-hot encoded labels
    
    noise_embedding = layers.Lambda(sinusoidal_embedding)(noise_variances)
    noise_embedding = layers.UpSampling2D(size=IMAGE_SIZE, interpolation="nearest")(
        noise_embedding
    )
    
    # Process class labels
    label_embedding = layers.Dense(32)(class_labels)
    label_embedding = layers.Reshape((1, 1, 32))(label_embedding)
    label_embedding = layers.UpSampling2D(size=IMAGE_SIZE, interpolation="nearest")(
        label_embedding
    )
    
    x = layers.Concatenate()([x, noise_embedding, label_embedding])
    
    skips = []
    
    x = DownBlock(32, block_depth=2)([x, skips])
    x = DownBlock(64, block_depth=2)([x, skips])
    x = DownBlock(96, block_depth=2)([x, skips])
    
    x = ResidualBlock(128)(x)
    x = ResidualBlock(128)(x)
    
    x = UpBlock(96, block_depth=2)([x, skips])
    x = UpBlock(64, block_depth=2)([x, skips])
    x = UpBlock(32, block_depth=2)([x, skips])
    
    x = layers.Conv2D(1, kernel_size=1, kernel_initializer="zeros")(x)
    unet = models.Model([noisy_images, noise_variances, class_labels], x, name="unet")

IMAGE_SIZE = 64
BATCH_SIZE = 64
DATASET_REPETITIONS = 5
LOAD_MODEL = False

NOISE_EMBEDDING_SIZE = 32
PLOT_DIFFUSION_STEPS = 20

EMA = 0.999
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 125

class DiffusionModel(models.Model):
    def __init__(self):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = unet
        self.ema_network = models.clone_model(self.network)
        self.diffusion_schedule = offset_cosine_diffusion_schedule

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = metrics.Mean(name="n_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker]

    def denormalize(self, images):
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, labels, training):
        if training:
            network = self.network
        else:
            network = self.ema_network
        pred_noises = network(
            [noisy_images, noise_rates**2, labels], training=training
        )
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, labels, diffusion_steps):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        
        for step in range(diffusion_steps):
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates, labels, training=False
            )
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            current_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
        return pred_images

    def generate(self, num_images, diffusion_steps, labels, initial_noise=None):
        if initial_noise is None:
            initial_noise = tf.random.normal(
                shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
            )
        generated_images = self.reverse_diffusion(
            initial_noise, labels, diffusion_steps
        )
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, data):
        images, labels = data #Unpack images and labels
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(tf.shape(images)[0], IMAGE_SIZE, IMAGE_SIZE, 1))

        diffusion_times = tf.random.uniform(
            shape=(tf.shape(images)[0], 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, labels, training=True
            )

            noise_loss = self.loss(noises, pred_noises)

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_weights)
        )

        self.noise_loss_tracker.update_state(noise_loss)

        for weight, ema_weight in zip(
            self.network.weights, self.ema_network.weights
        ):
            ema_weight.assign(EMA * ema_weight + (1 - EMA) * weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data #Unpack images and labels
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
        diffusion_times = tf.random.uniform(
            shape=(BATCH_SIZE, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, labels, training=False
        )
        noise_loss = self.loss(noises, pred_noises)
        self.noise_loss_tracker.update_state(noise_loss)

        return {m.name: m.result() for m in self.metrics}



def make_padding(data):
    res = tf.Variable(tf.zeros((64, 64)))
    res = np.float32(res)
    for i in range(len(data)-1):
        for j in range(len(data[0])):
            res[i, j+15] = data[i, j]
    return res
@tf.function
def preprocess(img, label):
    img = tf.cast(img, tf.float32)
    # Example: pad to [64, 64] size from any smaller size
    img = tf.image.resize_with_pad(img, target_height=64, target_width=64)
    # Or use tf.pad if you need exact placements:
    # img = tf.pad(img, paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0,0]])
    normalized_img = (img - global_min) / (global_max - global_min + 1e-6)
    return normalized_img, label

@tf.function
def preprocess(img, label):  # Function takes both image and label
    img = tf.cast(img, tf.float32)
    normalized_img = (img - global_min) / (global_max - global_min + 1e-6)
    return normalized_img, label  # Return both normalized image and label

@tf.function
def postprocess(generated_img):
    return generated_img * (global_max - global_min) + global_min

with gzip.open("../band_data/s10.pkl.gz", "rb") as f:
    data = pickle.load(f)

_, _, efdms = stft(data, fs=srate, window='hann', nperseg=256, noverlap=192, nfft = 512)

data = np.float32(np.abs(efdms))

data = data[:,:50,1::2,:,:]

#for
for i in range(19, 32):
    data_ch = data[:,:,i,:,:]
    data_ch = data_ch[:,:,:65,:]

    padded_data = []
    for k in range(2):
        padded_data.append([])
        for j in range(len(data_ch[LEFT])):
            padded_data[k].append([make_padding(data_ch[k, j])])

    padded_data = np.array(padded_data)
    padded_data = np.squeeze(padded_data)

    # Reshape data to (samples, height, width, 1)
    left_data = np.expand_dims(data_ch[0], axis=-1)  # Shape: (trials, height, width, 1)
    right_data = np.expand_dims(data_ch[1], axis=-1)  # Shape: (trials, height, width, 1)

    # Create labels
    left_labels = np.zeros(len(left_data))  # 0 for left hand
    right_labels = np.ones(len(right_data))  # 1 for right hand

    # Stack left and right trials into one dataset
    images = np.concatenate([left_data, right_data], axis=0)  # Shape: (2 * trials, height, width, 1)
    labels = np.concatenate([left_labels, right_labels], axis=0)  # Shape: (2 * trials,)

    # Convert labels to one-hot encoding
    labels_onehot = tf.one_hot(labels.astype(np.int32), 2)  # 2 classes: left and right

    # Create a TensorFlow dataset with both images and labels
    dataset = tf.data.Dataset.from_tensor_slices((images, labels_onehot))

    dataset = dataset.shuffle(len(images))  # Shuffle the dataset

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    dataset = dataset.cache()

    # Initialize global values
    global_min = float('inf')
    global_max = float('-inf')

    # Compute global min/max from your dataset
    for img, label in dataset:  # Unpacking image and label
        batch_min = tf.reduce_min(img)
        batch_max = tf.reduce_max(img)
        global_min = min(global_min, float(batch_min))
        global_max = max(global_max, float(batch_max))

    # Usage for preprocessing
    train = dataset.map(preprocess)
    train = train.repeat(80)
    train = train.batch(64, drop_remainder = True)


    with strategy.scope():
        ddm = DiffusionModel()
        ddm.normalizer.adapt(train.map(lambda x, y:x)) #Only adapt to images, not labels

        optimizer=optimizers.experimental.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        ddm.compile(
            #optimizer=optimizers.AdamW(
            optimizer=optimizer,
            loss=losses.MSE,
        )

    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath="./s34_class32_cp/checkpoint{:02}.ckpt".format(i),
        save_weights_only=True,
        save_freq="epoch",
        verbose=0,
    )
    
    ddm.fit(
        train,
        epochs=EPOCHS,
        callbacks=[model_checkpoint_callback] #image_generator_callback
    )


