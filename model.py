import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib as plt
import numpy as np

(ds_train, ds_test), ds_info = tfds.load('emnist/letters', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True)
print(ds_train)