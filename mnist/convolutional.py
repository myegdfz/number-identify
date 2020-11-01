import os

from mnist import input_data
from mnist import model
import tensorflow as tf

data = input_data.mnist.read_data_sets('MNIST_data', one_hot=True)
# define model
with tf.variable_scope('convolutional'):
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    keep_prob = tf.placeholder(tf.float32)
    y, variables = model.convolutional(x, keep_prob)

# train
_y = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(_y * tf.log(y))

print('cross_entropy: ')
print(cross_entropy)

learn_rate = 1e-4
train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(_y, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables)

with tf.Session() as sess:
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('mnist_log/1', sess.graph)
    summary_writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    # 断点续训
    ckpt = tf.train.get_checkpoint_state(
        os.path.join(os.path.dirname(__file__), 'MNIST_data/CkptData/', 'convolutional.cpkt')
    )
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    for _ in range(5000):
        batch = data.train.next_batch(50)
        if _ % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], _y: batch[1], keep_prob:1.0})
            print("step %d, train accuracy=%f"%(_, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], _y: batch[1], keep_prob:0.5})
    print(sess.run(accuracy, feed_dict={x: data.test.images, _y: data.test.labels, keep_prob: 1.0}))

    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'MNIST_data/CkptData', 'convolutional.ckpt'), write_meta_graph=False, write_state=False
    )
    print('Saved: ', path)