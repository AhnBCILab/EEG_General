import numpy as np

# mne imports
import mne
from mne import io
from mne.datasets import sample

# EEGNet-specific imports
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')

# bringing file
import pickle

with open("../test_folder2/gensig64.pickle", "rb") as f:
    data = pickle.load(f)

data = np.array(data)
data = np.squeeze(data)
data = data.transpose(0,2,1,3)

data.shape

left_labels = np.zeros(len(data[0]))  # 0 for left hand
right_labels = np.ones(len(data[0]))  # 1 for right hand

cdata = np.concatenate([data[0], data[1]], axis=0)  # Shape: (2 * trials, height, width, 1)
labels = np.concatenate([left_labels, right_labels], axis=0)  # Shape: (2 * trials,)

with open("../a.pickle", "rb") as f:
    datas = pickle.load(f)

sleft_labels = np.zeros(len(datas[0]))  # 0 for left hand
sright_labels = np.ones(len(datas[0]))  # 1 for right hand

scdata = np.concatenate([datas[0], datas[1]], axis=0)  # Shape: (2 * trials, height, width, 1)
slabels = np.concatenate([sleft_labels, sright_labels], axis=0)  # Shape: (2 * trials,)

accuracies = []

for i in range(0, 11):
    a = 0
    for j in range(0, 10):
    
        # Generate a permutation of indices
        indices = np.random.permutation(len(cdata))
        
        # Shuffle both arrays using the same indices
        X = cdata[indices] * 1000
        y = labels[indices]
        
        # Shuffle both arrays using the same indices
        sX = scdata[indices] * 1000
        
        kernels, chans, samples = 1, 64, 2304
    
        part_X = X[0:i*10]
        part_sX = sX[i*10:100]
        
        X_train      = np.concatenate([part_X, part_sX], axis = 0)
        Y_train      = y[0:100]
        X_validate   = sX[100:150,]
        Y_validate   = y[100:150]
        X_test       = sX[150:,]
        Y_test       = y[150:]
    
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
        
        # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
        # model configurations may do better, but this is a good starting point)
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
        fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 300, 
                                verbose = 2, validation_data=(X_validate, Y_validate),
                                callbacks=[checkpointer], class_weight = class_weights)
        
        # load optimal weights
        model.load_weights('/tmp/checkpoint.h5')
        
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
        a += acc    
    accuracies.append(a/10)


with open("./accuracies.pickle", "wb") as f:
    pickle.dump(accuracies, f)