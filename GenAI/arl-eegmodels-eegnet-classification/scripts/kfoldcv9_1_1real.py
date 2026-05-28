import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Prevents some memory allocator bugs

# EEGNet-specific imports
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import gc

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# tools for plotting confusion matrices
#from matplotlib import pyplot as plt

import pickle
import gzip

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# result = []
# result_sd = []
for i in range(18,53):
    with gzip.open("../band_data/s{:02}.pkl.gz".format(i), "rb") as f:
        data = pickle.load(f)
    
    data = data[:,:,1::2,:]
    
    data.shape
    left_labels = np.zeros(len(data[0]))  # 0 for left hand
    right_labels = np.ones(len(data[0]))  # 1 for right hand
    
    cdata = np.concatenate([data[0], data[1]], axis=0)  # Shape: (2 * trials, height, width, 1)
    labels = np.concatenate([left_labels, right_labels], axis=0)  # Shape: (2 * trials,)
    
    kernels, chans, samples = 1, 32, len(data[0][0][0])
    
    X = cdata * 1000
    y = labels

    y_cat = to_categorical(y)
    n_classes = y_cat.shape[1]
    
    for repeat in range(10):
        print(f"\n=== Repetition {repeat + 1} ===")
        all_accuracies = []
        
        for trial in range(10):
            print(f"Repetition {repeat + 1}, Trial {trial + 1}")
    
            # Split 10% test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y_cat, test_size=0.20, stratify=y_cat, random_state=repeat * 10 + trial
            )
    
            # Split 10% validation from remaining 90% → results in 80% train, 10% val
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=1/8, stratify=y_temp, random_state=repeat * 10 + trial
            )
    
            # Initialize EEGNet
            model = EEGNet(nb_classes=n_classes, Chans=chans, Samples=samples)
            model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    
            # Early stopping
            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    
            # Train
            model.fit(X_train, y_train,
                      validation_data=(X_val, y_val),
                      batch_size=1,
                      epochs=300,
                      callbacks=[es],
                      verbose=0)
    
            # Evaluate on test set
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"Test Accuracy: {acc:.4f}")
            all_accuracies.append(acc)
            
            # At the end of each trial
            K.clear_session()
            del model
            gc.collect()

        with open("./kfold_part_accx3/kfoldacc{:02}{:02}.pickle".format(i,repeat), "wb") as f:
            pickle.dump(all_accuracies, f)

    
    # # Final result
    # print("\n=== Final Results ===")
    # print(f"Average Test Accuracy (10x10): {np.mean(all_accuracies):.4f}")
    # print(f"Standard Deviation: {np.std(all_accuracies):.4f}")

    # with open("./real_kfold_acc/kfoldacc{:02}.pickle".format(i), "wb") as f:
    #     pickle.dump(np.mean(all_accuracies), f)

    # with open("./real_kfold_acc/kfoldsd{:02}.pickle".format(i), "wb") as f:
    #     pickle.dump(np.std(all_accuracies), f)
