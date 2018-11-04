import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pickle

training = pickle.load(open('train2.dat', 'rb'))
val = pickle.load(open('val2.dat', 'rb'))

#Preparing Loaded data
X_train = training['val']
Y_train = training['key']
X_val = val['val']
Y_val = val['key']



epochs = 50
n_classes = 2
batch_size = 32
chunk_size = 3
n_chunks = 2000
rnn_size = 60


x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output

def train_neural_network(x, X_train, Y_train, X_val, Y_val):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean( prediction - y)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  #initalizes the varibles in prep for training


        for e in range(0, epochs):
            epoch_loss = 0 
            #s = np.arange(X_train.shape[0])
            shuffle_set = np.random.permutation(np.arange(len(Y_train))) #randomly shuffles data for each epoch to combat overfitting
            #print(X_train.shape)
            #print(Y_train.shape)
            X_train = X_train[shuffle_set]
            Y_train = Y_train[shuffle_set]

            for i in range(0, len(Y_train) // batch_size):
                print("Batch: {0}".format(str(i) + "/" + str(len(Y_train) // batch_size)), end="\r")
                start = i * batch_size
                batch_x = X_train[start:start + batch_size]
                batch_y = Y_train[start:start + batch_size]
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y}) #runs the batch in the optimizer and gets the cost
                epoch_loss += c #compounds cost for epoch
            testing_acc = cost.eval({x: X_val, y: Y_val}) #evaluates accuracy of model on unseen data
            training_acc = cost.eval({x: X_train, y: Y_train}) #evaluates accracy of model on seen data

            print('[+] -----   Epoch: ', e, 'Accuracy on Unseen Data:', testing_acc, 'Accuracy on TRAINING data: ', training_acc, 'Training Loss: ', epoch_loss)
        preds = sess.run([prediction], feed_dict={x:X_val})

        for i in range(len(preds)):
            print(preds[i])
        print(Y_val)

train_neural_network(x, X_train, Y_train, X_val, Y_val)