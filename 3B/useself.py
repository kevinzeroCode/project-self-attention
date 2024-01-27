import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
import random

def load_image(path, size):
    image = cv2.imread(path)
    image = cv2.resize(image, (size,size))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   # shape: (size,size,3) -> (size,size,1)
    image = image/255.   # normalize
    return image

def load_data(size, root_path):
    images = []
    masks = []
    
    x = 0   # additional variable to identify images consisting of 2 or more masks
    
    for path in sorted(glob(root_path)):
        img = load_image(path, size)   # read mask or image
            
        if 'mask' in path:
            if x:   # this image has masks more than one
                masks[-1] += img   # add the mask to the last mask
                    
                # When 2 masks are added, the range can increase by 0-2. So we will reduce it again to the range 0-1.
                masks[-1] = np.array(masks[-1]>0.5, dtype='float64')
            else:
                masks.append(img)
                x = 1   # if the image has a mask again, the above code will run next time
        else:
            images.append(img)
            x = 0   # for moving to the next image
    return np.array(images), np.array(masks)


size = 128   # image size: 128x128
X, y = load_data(size, root_path = './Dataset_BUSI_with_GT/*/*')
X = X[:647]
y = y[:647]
X = np.expand_dims(X, -1)
y = np.expand_dims(y, -1)

import tensorflow as tf
from keras_self_attention import SeqSelfAttention  # 自定義的 SelfAttention 層
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras.models import Model

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(SelfAttention, self).__init__()
        self.filters = filters

        self.query = tf.keras.layers.Conv2D(filters // 8, kernel_size=1, strides=1, padding='same')
        self.key = tf.keras.layers.Conv2D(filters // 8, kernel_size=1, strides=1, padding='same')
        self.value = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same')

        self.softmax = tf.keras.layers.Softmax(axis=-1)
        
    def call(self, inputs):

        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        
        attention_map = tf.matmul(q, k, transpose_b=True)
        attention_map = self.softmax(attention_map)
        
        attention_out = tf.matmul(attention_map, v)
        return attention_out

    
# 註冊自定義層到 Keras 的自定義對象範圍中
with tf.keras.utils.custom_object_scope({'SelfAttention': SelfAttention}):
    # 加載模型的代碼
    model = tf.keras.models.load_model("selfyour_model.h5")
# 載入整個模型
# model = tf.keras.models.load_model("selfyour_model.h5")
i = random.randint(0, 647)
# predictions = model.predict(img)

plt.subplot(1, 3, 1)
plt.imshow(X[i], cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(y[i], cmap='gray')
plt.title('Original Mask')

# # 顯示模型預測結果
plt.subplot(1, 3, 3)
predicted_mask = model.predict(np.expand_dims(X[i],0),verbose=0)[0]
binary_mask = np.where(predicted_mask > 0.35, 1, 0)
plt.imshow(binary_mask, cmap='gray')
plt.title('Model Prediction')
plt.show()


