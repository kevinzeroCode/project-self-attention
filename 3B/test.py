import tensorflow as tf
print("Num gpu Available: ", len(tf.config.experimental.list_physical_devices('GPU')))