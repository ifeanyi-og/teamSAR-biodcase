from keras import Model, layers
from keras.src.applications.mobilenet import _conv_block, _depthwise_conv_block
from keras.src.callbacks import History, EarlyStopping, TensorBoard
from keras.src.metrics import AUC
from keras.models import Sequential
import tensorflow as tf

from paths import TENSORBOARD_LOGS_PATH
from config import Config


def create_model(input_shape=(2,)):
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(2, activation='linear'), # Single Layer Dense NN, for all intents and purposes mimics SVM
        layers.Softmax(),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  # SVM loss
        metrics=[AUC(curve='PR', name='average_precision')]
    )
    model.summary()
    return model

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