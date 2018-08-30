
'''
A sequence to sequence learning implementation for sorting a list of numbers.
reference: https://github.com/drforester/Sequence_to_Sequence_Sorting/blob/master/sort_seq.py
'''

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

n_steps = 15
n_input = 100
keep_prob = 0.5
n_layers = 2
n_hidden = 200
n_classes = 100
lr = 0.01
batch_size = 32

# encode a given integer sequence into RNN compatible format (one-hot representation)
def encode(X,seq_len, vocab_size):
    x = np.zeros((len(X),seq_len, vocab_size), dtype=np.float32)
    for ind,batch in enumerate(X):
        for j, elem in enumerate(batch):
            x[ind, j, elem] = 1
    return x

# generate a stream of inputs for training
def batch_gen(batch_size=32, seq_len=10, max_no=100):
    # Randomly generate a batch of integer sequences (X) and its sorted counterpart (Y)
    x = np.zeros((batch_size, seq_len, max_no), dtype=np.float32)
    y = np.zeros((batch_size, seq_len, max_no), dtype=np.float32)

    while True:
    # Generates a batch of input
        X = np.random.randint(max_no, size=(batch_size, seq_len))
        Y = np.sort(X, axis=1)

        for ind,batch in enumerate(X):
            for j, elem in enumerate(batch):
                x[ind, j, elem] = 1

        for ind,batch in enumerate(Y):
            for j, elem in enumerate(batch):
                y[ind, j, elem] = 1

        yield x, y
        x.fill(0.0)
        y.fill(0.0)

def BLSTM(x):

    def lstm_cell():
        lstm_cell = rnn.LSTMCell(n_hidden, initializer=tf.orthogonal_initializer())
        # lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
        return lstm_cell

    cell_fw = rnn.MultiRNNCell([lstm_cell() for _ in range(n_layers)], state_is_tuple=True)
    cell_bw = rnn.MultiRNNCell([lstm_cell() for _ in range(n_layers)], state_is_tuple=True)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, dtype=tf.float32)
    outputs = tf.concat(outputs, 2)
    fc_input = tf.reshape(outputs, [-1, n_hidden*2])
    logits = tf.contrib.layers.fully_connected(fc_input, n_classes, activation_fn=None)
    logits = tf.reshape(logits, [-1, n_steps, n_classes])
    return logits

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_steps, n_classes])
logits = BLSTM(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
y_pred = tf.nn.softmax(logits)
y_pred = tf.reshape(y_pred, [-1, n_classes])
y_true = tf.reshape(y, [-1, n_classes])
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
correct_prediction = tf.equal(tf.round(y_pred), tf.round(y_true))
all_labels_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
accuracy = tf.reduce_mean(all_labels_true)
init = tf.global_variables_initializer()
with tf.Session() as session:
    tf.global_variables_initializer().run()
    # for i in range(2000):
    for ind,(X,Y) in enumerate(batch_gen(batch_size, n_steps, n_input)):
        is_training = True
        _, batch_cost, batch_accuracy = session.run([optimizer, cost, accuracy],feed_dict={x: X, y : Y})
        print('Iteration:%d train loss:%f train accuracy:%f'%(ind,batch_cost,batch_accuracy))
        if ind % 100 == 0:
            test = np.random.randint(n_input, size=(1, n_steps))
            np_sorted = np.sort(test)[0]
            testX = encode(test, n_steps, n_input) 
            testY = encode(np.sort(test), n_steps, n_input)
            test_pred = session.run([y_pred],feed_dict={x: testX, y : testY})
            test_pred = np.argmax(test_pred, axis=2)[0]
            is_equal = np.array_equal(np_sorted, test_pred)
            if is_equal:
                print(colors.ok+'-----CORRECTLY SORTED-----'+colors.close)
            else:
                print(colors.fail+'-----incorrectly sorted-----'+colors.close)
            print(np_sorted, ': sorted by NumPy algorithm')
            print(test_pred, ': sorted by trained RNN')
            print("\n")

