import numpy as np
import numpy.matlib
import os

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Flatten, Conv2D, BatchNormalization, Lambda
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop

from keras import regularizers

import librosa
import librosa.display
import matplotlib.pyplot as plt

def load_data(root, num_data=None):
    
    labels = ["music", "talk"]
    groups = ["train", "validation"]

    data = dict()

    for label in labels:

        for group in groups:

            path = os.path.join(root, group, label)
            file_names = os.listdir(path)
            file_path = os.path.join(path, file_names[0])

            dim = np.load(file_path).shape
            num_files = len(file_names)
            print("Loading %d files from %s - %s..." % (num_files, label, group))

            data["%s-%s" % (label, group)] = np.zeros((num_files, dim[1], dim[0]), dtype=np.float32)
            
            for i, file_name in enumerate(file_names):

                file_path = os.path.join(path, file_name)
                data["%s-%s" % (label, group)][i, :, :] = np.load(file_path).T

    x_train = np.concatenate((data["music-train"], data["talk-train"]))
    y_train = np.concatenate((np.matlib.repmat(np.array((1,0)), data["music-train"].shape[0], 1),
        np.matlib.repmat(np.array((0,1)), data["talk-train"].shape[0], 1)))

    x_val = np.concatenate((data["music-validation"], data["talk-validation"]))
    y_val = np.concatenate((np.matlib.repmat((1,0), data["music-validation"].shape[0], 1),
        np.matlib.repmat((0,1), data["talk-validation"].shape[0], 1)))

    return (x_train, y_train), (x_val, y_val)

def build_model(weights=None, stateful=False):

    # Parameters
    N_LAYERS = 3
    N_CLASSES = 2
    N_FEATURES = 128
    FILTER_LENGTH = 5
    CONV_FILTER_COUNT = 56
    LSTM_COUNT = 96
    NUM_HIDDEN = 64
    L2_regularization = 0.001

    input_shape = (None, N_FEATURES)
    model_input = Input(input_shape, name='input')

    print('Building model...')
    layer = model_input
    
    ### 3 1D Convolution Layers
    for i in range(N_LAYERS):
        # give name to the layers
        layer = Conv1D(
                filters=CONV_FILTER_COUNT,
                kernel_size=FILTER_LENGTH,
                kernel_regularizer=regularizers.l2(L2_regularization),  # Tried 0.001
                name='convolution_' + str(i + 1)
            )(layer)
        layer = BatchNormalization(momentum=0.9)(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling1D(2)(layer)
        layer = Dropout(0.4)(layer)
    
    ## LSTM Layer
    layer = LSTM(LSTM_COUNT, return_sequences=False, stateful=stateful)(layer)
    layer = Dropout(0.4)(layer)
    
    ## Dense Layer
    layer = Dense(NUM_HIDDEN, kernel_regularizer=regularizers.l2(L2_regularization), name='dense1')(layer)
    layer = Dropout(0.4)(layer)
    
    ## Softmax Output
    layer = Dense(N_CLASSES)(layer)
    layer = Activation('softmax', name='output_realtime')(layer)
    model_output = layer
    model = Model(model_input, model_output)
    
    if weights:
        model.set_weights(weights)
    
    opt = Adam(lr=0.001)
    model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )

    return model

def train_model(x_train, y_train, x_val, y_val):

    # Parameters 
    BATCH_SIZE = 32
    EPOCH_COUNT = 10
    
    model = build_model(x_train.shape)

    checkpoint_callback = ModelCheckpoint('models/crnn/model_weights.h5', monitor='val_accuracy', verbose=1,
                                          save_best_only=True, mode='max')
    
    reducelr_callback = ReduceLROnPlateau(
                monitor='val_accuracy', factor=0.5, patience=10, min_delta=0.01,
                verbose=1
            )
    callbacks_list = [checkpoint_callback, reducelr_callback]

    # Fit the model and get training history.
    print('Training...')
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,
                        validation_data=(x_val, y_val), verbose=1, callbacks=callbacks_list)

    return model, history

def show_summary_stats(history):
    # List all data in history
    print(history.history.keys())

    # Summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == '__main__':

    data_root = "data"

    (x_train, y_train), (x_val, y_val) = load_data(data_root)
    print(x_train.shape)

    model, history = train_model(x_train, y_train, x_val, y_val)
    # Save the model architecture
    with open('models/crnn/model_architecture.json', 'w') as f:
        f.write(model.to_json())
    
    show_summary_stats(history)