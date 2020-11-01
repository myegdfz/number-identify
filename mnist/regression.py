import os

from mnist import input_data
from mnist import model
import tensorflow as tf

data = input_data.mnist.read_data_sets('MNIST_data', one_hot=True)

# create model
with tf.variable_scope("regression"):
    x = tf.placeholder(tf.float32, [None, 784])
    y, variables = model.regression(x)

# train
_y = tf.placeholder("float", [None, 10])
# 交叉熵
cross_entropy = -tf.reduce_sum(_y * tf.log(y))
# 训练步骤
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 预测
correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(_y, 1))
# 准确率
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

# 保存
saver = tf.train.Saver(variables)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        batch_xs, batch_ys = data.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, _y: batch_ys})

    print((sess.run(accuracy, feed_dict={x: data.test.images, _y: data.test.labels})))

    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'MNIST_data/CkptData','regression.ckpt'),
        write_meta_graph=False, write_state=False
    )
    print("Saved:", path)