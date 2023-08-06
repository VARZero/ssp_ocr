import tensorflow as tf
import tensorflow_datasets as tfds

ds, info = tfds.load('emnist', split='train', with_info=True)

print(info.splits['train'])

#for data in ds.take()


