# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
import numpy as np
from flask import Flask, jsonify, render_template, request
from keras.models import load_model
import matplotlib.pyplot as plt
import base64
from io import BytesIO

model_name = './mnist/other/models/GAN-CNN.h5'
base_model = load_model(model_name)
base_model.summary()


# tf.compat.v1.disable_eager_execution()
# x = tf.compat.v1.placeholder("float", [None, 784])
# sess = tf.compat.v1.Session()
#
# with tf.compat.v1.variable_scope("regression"):
#     _yr, variables = model.regression(x)
#
# saver = tf.compat.v1.train.Saver(variables)
# saver.restore(sess, "mnist/MNIST_data/CkptData/regression.ckpt")
#
# with tf.compat.v1.variable_scope("convolutional"):
#     keep_prob = tf.compat.v1.placeholder("float")
#     _yc, variables = model.convolutional(x, keep_prob)
#
# sess.run(tf.compat.v1.global_variables_initializer())
# saver = tf.compat.v1.train.Saver(variables)
# saver.restore(sess, "mnist/MNIST_data/CkptData/convolutional.ckpt")
#
#
# def regression(inx):
#     return sess.run(_yr, feed_dict={x: inx}).flatten().tolist()
#
#
# def convolutional(inx):
#     return sess.run(_yc, feed_dict={x: inx, keep_prob: 1.0}).flatten().tolist()  #


def cgenerator(inx):
    inx = inx.astype('float32')
    inx = np.reshape(inx, (1, 28, 28, 1))
    # return test_gen.get_function(inx)
    r_test = np.random.sample(inx.shape[0])
    pred = base_model.predict([inx, r_test])
    return pred[0], pred[1]


app = Flask(__name__)


@app.route('/api/mnist', methods=['post'])
def mnist():
    _input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 28, 28)
    # print(_input)
    _input[_input <= 0.08] = 0.
    _input[_input > 0.08] = 1.
    _output, img = cgenerator(_input)

    plt.figure(figsize=(10, 5), dpi=100)
    ax = plt.subplot(2, 1, 1)
    plt.imshow(_input.reshape(28, 28))
    plt.gray()
    ax.set_axis_off()
    ax = plt.subplot(2, 1, 2)
    plt.title('%s' % np.argmax(_output))
    plt.imshow(img.reshape(28, 28))
    plt.gray()
    ax.set_axis_off()
    plt.show()

    print("_output", _output[0].tolist())
    return jsonify(results=[_output[0].tolist()])


@app.route('/')
def main():
    return render_template('index.html')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)
