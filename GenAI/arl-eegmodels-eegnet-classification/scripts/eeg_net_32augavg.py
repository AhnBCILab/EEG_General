import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# EEGNet-specific imports
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')

# bringing file
import pickle
import gzip

with gzip.open("../band_data/s01.pkl.gz", "rb") as f:
    data = pickle.load(f)

data = data[:,:,1::2,:]

data_tr = data[:,:50,:,:]
data_te = data[:,50:,:,:]

with open("../test_folder2/generated_data32/s01gensig32.pickle", "rb") as f:
    gen_data = pickle.load(f)

gen_data = np.array(gen_data)
gen_data = np.squeeze(gen_data)
gen_data = gen_data.transpose(0,2,1,3)

gen_data = gen_data[:,:50,:,:]

# Combine data and labels as already done
cdata_tr = np.concatenate([data_tr[0], data_tr[1]], axis=0)
cdata_te = np.concatenate([data_te[0], data_te[1]], axis=0)
labels_tr = np.concatenate([np.zeros(len(data_tr[0])), np.ones(len(data_tr[1]))], axis=0)
labels_te = np.concatenate([np.zeros(len(data_te[0])), np.ones(len(data_te[1]))], axis=0)

indices = np.random.permutation(len(cdata_tr))

# Scale data
X_tr = cdata_tr[indices] * 1000
X_te = cdata_te[indices] * 1000
y_tr = labels_tr[indices]
y_te = labels_te[indices]

# Specify model input shapes
kernels, chans, samples = 1, 32, 2304

# Combine data and labels as already done
gcdata = np.concatenate([gen_data[0], gen_data[1]], axis=0)
glabels = np.concatenate([np.zeros(len(gen_data[0])), np.ones(len(gen_data[1]))], axis=0)

# Scale data
gX = gcdata[indices] * 1000
gy = glabels[indices]

gpus = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {gpus}")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using MirroredStrategy with {len(gpus)} GPUs")
else:
    strategy = tf.distribute.get_strategy()  # Use default strategy (CPU or single GPU)
    print("Using default strategy (CPU or single GPU)")

for j in range(0, 10):
    accuracies = []
    for i in range(0, 11): 
        part_X = gX[0:i*10]
        part_y = gy[0:i*10]
        #part_sX = X_tr[i*10:100]#r
        
        #X_train      = np.concatenate([part_X, part_sX], axis = 0)#r
        X_train      = np.concatenate([part_X, X_tr], axis = 0)
        #Y_train      = y_tr[0:100]#r
        Y_train      = np.concatenate([part_y, y_tr], axis = 0)
        X_validate   = X_te[0:50,]
        Y_validate   = y_te[0:50]
        X_test       = X_te[50:,]
        Y_test       = y_te[50:]
    
        ############################# EEGNet portion ##################################
    
        # convert labels to one-hot encodings.
        Y_train      = np_utils.to_categorical(Y_train)
        Y_validate   = np_utils.to_categorical(Y_validate)
        Y_test       = np_utils.to_categorical(Y_test)
        
        # convert data to NHWC (trials, channels, samples, kernels) format. Data 
        # contains 60 channels and 151 time-points. Set the number of kernels to 1.
        X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
        X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
        X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)
           
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(128).cache().repeat(5).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices((X_validate, Y_validate)).batch(128).cache().repeat(5).prefetch(tf.data.AUTOTUNE)

        
        # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
        # model configurations may do better, but this is a good starting point)
        with strategy.scope():
            model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples, 
                           dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                           dropoutType = 'Dropout')
            
            # compile the model and set the optimizers
            model.compile(loss='categorical_crossentropy', optimizer='adam', 
                          metrics = ['accuracy'])
            
            # count number of parameters in the model
            numParams    = model.count_params()    
        
        # set a valid path for your system to record model checkpoints
        checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                       save_best_only=True)
        
        ###############################################################################
        # if the classification task was imbalanced (significantly more trials in one
        # class versus the others) you can assign a weight to each class during 
        # optimization to balance it out. This data is approximately balanced so we 
        # don't need to do this, but is shown here for illustration/completeness. 
        ###############################################################################
        
        # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
        # the weights all to be 1
        class_weights = {0:1, 1:1}#, 2:1, 3:1}
        
        ################################################################################
        # fit the model. Due to very small sample sizes this can get
        # pretty noisy run-to-run, but most runs should be comparable to xDAWN + 
        # Riemannian geometry classification (below)
        ################################################################################
        # fittedModel = model.fit(X_train, Y_train, batch_size = 64, epochs = 300, 
        #                         verbose = 2, validation_data=(X_validate, Y_validate),
        #                         callbacks=[checkpointer], class_weight = class_weights)

        early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        fittedModel = model.fit(train_ds, epochs=300, validation_data=val_ds, callbacks=[checkpointer, early_stopper], class_weight=class_weights)

        
        # load optimal weights
        #model.load_weights('/tmp/checkpoint.h5')
        
        ###############################################################################
        # can alternatively used the weights provided in the repo. If so it should get
        # you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
        # system.
        ###############################################################################
        
        # WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5 
        # model.load_weights(WEIGHTS_PATH)
        
        ###############################################################################
        # make prediction on test set.
        ###############################################################################
        
        probs       = model.predict(X_test)
        preds       = probs.argmax(axis = -1)  
        acc         = np.mean(preds == Y_test.argmax(axis=-1))
        print("Classification accuracy: %f " % (acc))
        accuracies.append(acc)
    
    with open("./eeg_accuracies/acc01{:02}add_new.pickle".format(j), "wb") as f:
        pickle.dump(accuracies, f)
