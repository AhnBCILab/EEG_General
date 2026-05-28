import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# EEGNet-specific imports
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

import pickle
import gzip

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')

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


#result = []
#result_sd = []
for i in range(9,53):
    with gzip.open("../band_data/s{:02}.pkl.gz".format(i), "rb") as f:
        data = pickle.load(f)
    
    data = data[:,:,1::2,:2304]
    
    data.shape
    left_labels = np.zeros(len(data[0]))  # 0 for left hand
    right_labels = np.ones(len(data[0]))  # 1 for right hand
    
    cdata = np.concatenate([data[0], data[1]], axis=0)  # Shape: (2 * trials, height, width, 1)
    labels = np.concatenate([left_labels, right_labels], axis=0)  # Shape: (2 * trials,)
    
    kernels, chans, samples = 1, 32, 2304
    
    X = cdata * 1000
    y = labels
    
    y_cat = to_categorical(y)
    n_classes = y_cat.shape[1]
    
    # Settings
    num_repeats = 10
    num_folds = 10
    all_accuracies = []
    
    for repeat in range(num_repeats):
        print(f"\n=== Repetition {repeat + 1} ===")
        
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=repeat)
        fold_accuracies = []
        
        for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\nRepetition {repeat + 1}, Fold {fold + 1}")
            
            # Outer split
            X_train_val, X_test = X[train_val_idx], X[test_idx]
            y_train_val, y_test = y_cat[train_val_idx], y_cat[test_idx]
    
            # Inner validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.2, random_state=fold + repeat * 10, stratify=y_train_val
            )

            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).cache().prefetch(tf.data.AUTOTUNE)
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32).cache().prefetch(tf.data.AUTOTUNE)

            with strategy.scope():
                # Initialize EEGNet
                model = EEGNet(nb_classes=n_classes, Chans=chans, Samples=samples)
                model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
        
                # Early stopping
                es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    
            # Train model
            model.fit(train_ds,
                      validation_data=val_ds,
                      #batch_size=16,
                      epochs=100,
                      callbacks=[es],
                      verbose=0)
    
            # Evaluate
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"Test Accuracy: {acc:.4f}")
            fold_accuracies.append(acc)
        
        # Store fold results of one repetition
        all_accuracies.extend(fold_accuracies)
    
    # Final results
    print("\n=== Final Results ===")
    print("Average Accuracy across 10x10 CV: {:.4f}".format(np.mean(all_accuracies)))
    print("Standard Deviation: {:.4f}".format(np.std(all_accuracies)))

    with open("./kfold_acc/kfoldacc{:02}.pickle".format(i), "wb") as f:
        pickle.dump(np.mean(all_accuracies), f)

    with open("./kfold_acc/kfoldsd{:02}.pickle".format(i), "wb") as f:
        pickle.dump(np.std(all_accuracies), f)

    # result.append(np.mean(all_accuracies))
    # result_sd.append(np.std(all_accuracies))

# result = np.array(result)
# result_sd = np.array(result_sd)

# with open("./kfoldresult.pickle", "wb") as f:
#     pickle.dump(result, f)

# with open("./kfoldresult_sd.pickle", "wb") as f:
#     pickle.dump(result_sd, f)
    