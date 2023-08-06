# %%
import tensorflow as tf
import tensorflow_datasets as tfds


# %% [markdown]
# 데이터 셋 로드 (TFDS에서 가져온 EMNIST)

# %%
(etfds_train, etfds_test), etfds_info = tfds.load("emnist", split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True)
print(etfds_info)
print(etfds_train)

def normalize_img_ascii_lable(image, label):
    ascii_Lb = label
    if ascii_Lb >= 10:
        ascii_Lb += 8
    elif ascii_Lb >= 43:
        ascii_Lb += 6
    ascii_Lb += 15
    image = tf.transpose(image)
    return tf.cast(image, tf.float32) / 255., ascii_Lb

etfds_train = etfds_train.map(normalize_img_ascii_lable, num_parallel_calls=tf.data.experimental.AUTOTUNE)
etfds_train = etfds_train.cache()
etfds_train = etfds_train.shuffle(etfds_info.splits['train'].num_examples)
etfds_train = etfds_train.batch(128)
etfds_train = etfds_train.prefetch(tf.data.experimental.AUTOTUNE)
print(etfds_train)

etfds_test = etfds_test.map(normalize_img_ascii_lable, num_parallel_calls=tf.data.experimental.AUTOTUNE)
etfds_test = etfds_test.batch(128)
etfds_test = etfds_test.cache()
etfds_test = etfds_test.prefetch(tf.data.experimental.AUTOTUNE)