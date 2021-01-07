from tensorflow import keras
import sys
from keras.datasets import mnist
from keras.models import load_model
from keras.models import Model
from keras.models import Input
import numpy as np
import matplotlib.pyplot as plt

model_name = './models/GAN-CNN.h5'
base_model = load_model(model_name)
base_model.summary()

layers = [o for o in base_model.layers]

data_delta = np.load("test_mnist.npz")
features = data_delta['features']
labels = data_delta['labels']
print(features.astype('float32'))
features = features.astype('float32')/255
print(features)
features = np.reshape(features, (len(features), 28, 28, 1))


inputs = Input(shape=(10, ))
random_input = Input(shape=(1, ))
print(inputs)

net = None
for i in range(13, len(layers)):
    print(layers[i].name)
    if i == 13:
        net = layers[i]([inputs, random_input])
    else:
        net = layers[i](net)

r_test = np.random.sample(features.shape[0])
print('r_test.shape', r_test.shape)
model = Model(inputs=[inputs, random_input], outputs=net)
model.summary()

pred = base_model.predict([features, r_test])
pred_y = pred[0]
decoded_img = pred[1]
print('pred_y', pred_y.shape)
_sum = 0.0
for i in range(features.shape[0]):
    _sum += (labels[i] == np.argmax(pred_y[i]))

print('accuracy: %f' % (100. * _sum/features.shape[0]))



n = 10
plt.figure(figsize=(10, 5), dpi=100)
for i in range(n):
    # display original pic
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(features[i].reshape(28, 28))
    plt.gray()
    ax.set_axis_off()

    # display reconstruction
    ax = plt.subplot(2, n, i+n+1)
    plt.title('%s' % np.argmax(pred_y[i]))
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.set_axis_off()

plt.show()