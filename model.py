import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_bilstm(input_shape, n_classes, conv_filters=64, lstm_units=128,
                     dropout_rate=0.3):
    """
    Builds SM-CBNet.

    Parameters
    ----------
    input_shape : tuple
        Shape of a single sample excluding batch dim, e.g. (timesteps, features).
    n_classes : int
        Number of target classes.
    conv_filters : int, default 64
        Number of filters for Conv1D.
    lstm_units : int, default 128
        Units for BiLSTM.
    dropout_rate : float, default 0.3
        Dropout rate between layers.

    Returns
    -------
    tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(conv_filters, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(x)
    x = layers.Dropout(dropout_rate)(x)

    if n_classes == 2:
        activation = 'sigmoid'
        units = 1
        loss = 'binary_crossentropy'
    else:
        activation = 'softmax'
        units = n_classes
        loss = 'sparse_categorical_crossentropy'

    outputs = layers.Dense(units, activation=activation)(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=loss,
        metrics=['accuracy']
    )
    return model