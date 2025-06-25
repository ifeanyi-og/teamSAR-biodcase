from keras import Model, layers
from keras.src.applications.mobilenet import _conv_block, _depthwise_conv_block
from keras.src.callbacks import History, EarlyStopping, TensorBoard
from keras.src.metrics import AUC
import tensorflow as tf
import numpy as np

from paths import TENSORBOARD_LOGS_PATH
from config import Config

from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.losses import hinge

import tensorflow as tf

#''' Ifeanyichukwu, don't fumble the bag twin
def create_model(spec_shape, n_filters_1=20, n_filters_2=32, dropout=0.4) -> Model:
    inputs = layers.Input(shape=spec_shape, name = "mel_input")
    x = _conv_block(inputs, filters=n_filters_1, alpha=1, kernel=(10, 4), strides=(5, 2))
    # x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_2, alpha=1, block_id=1)
    x = layers.GlobalMaxPooling2D(keepdims=True)(x)
    x = layers.Dropout(dropout, name="dropout1")(x)
    x = layers.Flatten()(x)
    # print("x flattened shape is", x.shape)
    # input2 = layers.Input(shape=corr_shape, name = "autocorr_input")
    # y = layers.Dense(32, activation="relu", name="autocor_dense")(input2)
    # combined = layers.Concatenate()([x, y])
    # combined = x
    # combined = layers.Dense(64, activation="relu")(combined)
    # combined = layers.Dropout(dropout)(combined)
    # x = layers.Dense(2, activation="softmax")(x)

    # Killed depthwise conv block
    # dropped number of filters in conv layer to 20
    # Dropout to 0.4
    # Restore Best Weights = True

    x = layers.Dense(2)(x)
    outputs = layers.Softmax()(x)
    model = Model(inputs, outputs, name="mobilenet_slimmed")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision')]
    )
    model.summary()
    return model


'''
def create_model(input_shape=(6,)):
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(1, activation='linear')  # No activation; treat it as SVM
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  # SVM loss
        metrics=[AUC(curve='PR', name='average_precision')]
    )
    model.summary()
    return model
'''

def train_model(model: Model, train_ds, valid_ds, config: Config, class_weight) -> Model:
    tr_cfg = config.model_training

    train_ds = train_ds.cache().shuffle(tr_cfg.shuffle_buff_n).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.cache().prefetch(tf.data.AUTOTUNE)
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=tr_cfg.n_epochs,
        class_weight=class_weight,
        callbacks=[
            EarlyStopping(
                patience=tr_cfg.early_stopping.patience,
                restore_best_weights=True,
            ),
            TensorBoard(TENSORBOARD_LOGS_PATH, update_freq=1)
        ]
    )
    return model