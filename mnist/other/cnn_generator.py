from tensorflow import keras
import datetime
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, GlobalMaxPooling2D, Dense, Reshape, Flatten, \
    Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# create model
input_img = Input(shape=(28, 28, 1))
# encoder
x = Conv2D(filters=32, kernel_size=3, padding='same', strides=1, activation='relu')(input_img)
x = Conv2D(filters=32, kernel_size=3, padding='same', strides=1, activation='relu')(x)
x = Conv2D(filters=32, kernel_size=3, padding='same', strides=2, activation='relu')(x)

x = Conv2D(filters=16, kernel_size=3, padding='same', strides=1, activation='relu')(x)
x = Conv2D(filters=16, kernel_size=3, padding='same', strides=1, activation='relu')(x)
x = Conv2D(filters=16, kernel_size=3, padding='same', strides=2, activation='relu')(x)


x = Conv2D(filters=8, kernel_size=3, padding='same', strides=1, activation='relu')(x)
x = Conv2D(filters=8, kernel_size=3, padding='same', strides=1, activation='relu')(x)
x = Conv2D(filters=8, kernel_size=3, padding='same', strides=2, activation='relu')(x)

x = GlobalMaxPooling2D()(x)
encoded = Dense(10, activation='softmax', name='encoded')(x)
random_input = Input(shape=(1, ))
print('encoded', encoded)
concat = Concatenate(axis=1)([encoded, random_input])

print('concat[0].shape', concat[0].shape)

x = Dense(16, activation='relu')(concat)
x = Dense(32, activation='relu')(x)

# reshape
x = Reshape((1, 1, 32))(x)

# decoder
x = Conv2DTranspose(32, kernel_size=3, padding='same', strides=3, activation='relu')(x)
x = Conv2DTranspose(16, kernel_size=3, padding='valid', strides=2, activation='relu')(x)
x = Conv2DTranspose(16, kernel_size=3, padding='same', strides=2, activation='relu')(x)
x = Conv2DTranspose(8, kernel_size=3, padding='same', strides=2, activation='relu')(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoded')(x)

auto_encoder = Model([input_img, random_input], [encoded, decoded])

auto_encoder.compile(
    optimizer=keras.optimizers.Adam(), 
    loss=['categorical_crossentropy', 'binary_crossentropy'],
    loss_weights=[0.1, 1],
    metrics=['accuracy']
)

auto_encoder.summary()
keras.utils.plot_model(auto_encoder, to_file='model.png')

# load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data('./mnist.npz')

print(x_train.shape)

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

print('x_test.shape', x_test.shape)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print('y_train.shape', y_train.shape)
print('y_test.shape', y_test.shape)

r_train = np.random.sample(x_train.shape[0])
r_test = np.random.sample(x_test.shape[0])

print('r_train.shape', r_train.shape)
print('r_test.shape', r_test.shape)

# tensorboard --logdir=logs
auto_encoder.fit([x_train, r_train], [y_train, x_train], epochs=1, batch_size=128, shuffle=True,
                 validation_data=([x_test, r_test], [y_test, x_test]), verbose=1,
                 callbacks=[
                     TensorBoard(log_dir='logs'),
                     ModelCheckpoint('./models/%s.h5' % 'GAN-g', monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
                 ])


# api for flask( return the possible rate) (1*10)
def cnn_generator(idx):
    print(idx.shape)
    _r_test = np.random.sample(idx.shape[0])
    print(_r_test)
    _pred = auto_encoder.predict([idx, _r_test])
    return _pred[0]


print('after x_test', x_test.shape)
# take a look at the reconstructed digits
pred = auto_encoder.predict([x_test, r_test])
pred_y = pred[0]    # fake label
decoded_img = pred[1]  # fake img

print(pred_y)
n = 10
plt.figure(figsize=(10, 5), dpi=100)
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.set_axis_off()

    # display reconstruction
    ax = plt.subplot(2, n, i+n+1)
    plt.title('%s' % np.argmax(pred_y[i]))
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.set_axis_off()

plt.show()