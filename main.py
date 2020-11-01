# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from mnist import model

x = tf.placeholder("float", [None, 784])
sess = tf.Session()

with tf.variable_scope("regression"):
    _yr, variables = model.regression(x)

saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/MNIST_data/CkptData/regression.ckpt")

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    _yc, variables = model.convolutional(x, keep_prob)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/MNIST_data/CkptData/convolutional.ckpt")


def regression(inx):
    return sess.run(_yr, feed_dict={x: inx}).flatten().tolist()


def convolutional(inx):
    return sess.run(_yc, feed_dict={x: inx, keep_prob: 1.0}).flatten().tolist()


app = Flask(__name__)


@app.route('/api/mnist', methods=['post'])
def mnist():
    # print(np.array(request.json, dtype=np.uint8))
    _input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    _output_regression = regression(_input)
    _output_convolutional = convolutional(_input)
    return jsonify(results=[_output_regression, _output_convolutional])


@app.route('/')
def main():
    return render_template('index.html')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)