import numpy as np
import tensorflow as tf
from mnist import model
from PIL import Image
import os

tf.reset_default_graph()
x = tf.placeholder("float", [None, 784])
sess = tf.Session()

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    _yc, variables = model.convolutional(x, keep_prob)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/MNIST_data/CkptData/convolutional.ckpt")


def convolutional(inx):
    return sess.run(_yc, feed_dict={x: inx, keep_prob: 1.0}).flatten().tolist()


# open picture
def pre_pic(picture_path):
    img = Image.open(picture_path)
    re_img = img.resize((28, 28), Image.ANTIALIAS)
    # transfer into Grey graph
    # img_grey = np.array(re_img.convert('L'))
    _input = ((255 - np.array(re_img, dtype=np.uint8)) / 255.0).reshape(1, 784)
    _output_convolutional = convolutional(_input)
    return _output_convolutional


# input test dataSet
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def application():
    dataset_dir = r'E:\Image Identification\DataSet\OwnMade\ceshi'
    image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]
    for image_filename in image_filenames:
        print(image_filename, '\t', np.argmax(pre_pic(image_filename)))


if __name__ == '__main__':
    application()
