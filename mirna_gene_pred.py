import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

data = 'mirna_gene_dataset.csv' #input data

global count_i
count_i = 0
global count_i2
count_i2 = 0


tr_iter = 5000000

batch_size = 256

n_inputs = 16
n_steps = 16
n_hidden_units = 128
n_classes = 2

x = tf.compat.v1.placeholder (tf.float32, [None, n_steps, n_inputs])
y = tf.compat.v1.placeholder (tf.float32, [None, n_classes])

weights = {
    # (28, 128)
    'in': tf.Variable(tf.random.normal([n_inputs, n_hidden_units])),
    # matrix(128, 10)
    'out': tf.Variable(tf.random.normal([n_hidden_units, n_classes]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

lr = 0.001

def RNN(X, weights, biases):

    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])


    cell = tf.compat.v1.keras.layers.LSTMCell(n_hidden_units)
    print(cell.output_size, cell.state_size)
    outputs, final_state = tf.nn.dynamic_rnn(cell,X_in,dtype=tf.float32)

    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


def batches(batch_size, features, labels):
    global list_bx
    global list_by
    assert len(features) == len(labels)
    output_batches = []
    list_bx = []
    list_by = []
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch_x = features[start_i:end_i]
        list_bx.append(batch_x)
        batch_y = labels[start_i:end_i]
        list_by.append(batch_y)


def next_batches(i):
    global list_bx
    global list_by
    global count_i
    if count_i < len(list_bx) - 2:
        batch_x = list_bx[count_i]
        batch_y = list_by[count_i]
        count_i += 1
    else:
        batch_x = list_bx[count_i]
        batch_y = list_by[count_i]
        count_i = 0
    batch_x = batch_x.values
    batch_y = batch_y.values

    return batch_x, batch_y


def next_test_batches(i):
    global test_bx
    global test_by
    global count_i2
    if count_i2 < len(test_bx) - 2:
        batch_x = test_bx[count_i2]
        batch_y = test_by[count_i2]
        count_i2 += 1
    else:
        batch_x = test_bx[count_i2]
        batch_y = test_by[count_i2]
        count_i2 = 0
    batch_x = batch_x.values
    batch_y = batch_y.values

    return batch_x, batch_y


def test_batches(batch_size, features, labels):
    global test_bx
    global test_by
    assert len(features) == len(labels)
    output_batches = []
    test_bx = []
    test_by = []
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch_x = features[start_i:end_i]
        test_bx.append(batch_x)
        batch_y = labels[start_i:end_i]
        test_by.append(batch_y)



try:
    df = pd.read_csv(data)
    label = df['label']
    df = df.drop(['0_mirna'], axis=1)
    df = df.drop(['1_gene'], axis=1)
    df = df.drop(['label'], axis=1)
    list_label2 = []
except:
    print('error data')

#gene miRNA 관계여부 label
for i in range(0, len(label)):
    if label[i] == 1:
        list_label2.append(0)
    else:
        list_label2.append(1)

list_2_pd = pd.DataFrame(list_label2)

#label = abx_label + 반대되는 abx_label
label = pd.concat([label, list_2_pd], axis=1)

X = df
Y = label

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
batch_size = 64

test_batches(batch_size,X_test,y_test)
batches(batch_size, X_train, y_train)

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < tr_iter:

        batch_xs, batch_ys = next_batches(step)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])

        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })

        if step % 500 == 0:
            print('training set acc', step)
            print(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,
            }))
            print('test set acc', step)
            x_test_batch, y_test_batch = next_test_batches(step)
            x_test_batch = x_test_batch.reshape([batch_size, n_steps, n_inputs])
            print(sess.run(accuracy, feed_dict={
                x: x_test_batch,
                y: y_test_batch,

            }))

        step += 1


