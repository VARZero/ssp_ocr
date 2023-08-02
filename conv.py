import tensorflow as tf
import tensorflowjs as tfjs

cnnEmnist = tf.keras.models.load_model('./model/spemnist.h5')
tfjs.converters.save_keras_model(cnnEmnist, './model/cnnemnistjs')