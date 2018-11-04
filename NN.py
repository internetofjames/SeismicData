import tensorflow as tf
import numpy as np
import pickle

training = pickle.load(open('train5.dat', 'rb'))
val = pickle.load(open('val5.dat', 'rb'))

#Preparing Loaded data
X_train = training['val']
Y_train = training['key']
X_val = val['val']
Y_val = val['key']

#defining layer paramerters
inputs = 3000
layer_1_neurons = 12
layer_2_neurons = 8
layer_3_neurons = 6
output_neurons = 2

learning_rate = 0.00001
epochs = 50000
batch_size = 8

#Creating place holders for inputs and outputs
X = tf.placeholder(dtype=tf.float32, shape=[None, inputs])
Y = tf.placeholder(dtype=tf.float32, shape=[None, output_neurons])

#defining layer attributes
h_1_layer_weights = tf.Variable(tf.random_normal([inputs, layer_1_neurons]))
h_1_layer_bias = tf.Variable(tf.random_normal([layer_1_neurons]))

h_2_layer_weights = tf.Variable(tf.random_normal([layer_1_neurons, layer_2_neurons]))
h_2_layer_bias = tf.Variable(tf.random_normal([layer_2_neurons]))

h_3_layer_weights = tf.Variable(tf.random_normal([layer_2_neurons, layer_3_neurons]))
h_3_layer_bias = tf.Variable(tf.random_normal([layer_3_neurons]))

output_weights = tf.Variable(tf.random_normal([layer_3_neurons, output_neurons]))
output_bias = tf.Variable(tf.random_normal([output_neurons]))
#defining layer calculators and flow
#REMEMBER Weights
l1_calc = tf.nn.relu(tf.add(tf.matmul(X, h_1_layer_weights), h_1_layer_bias))
l2_calc = tf.nn.relu(tf.add(tf.matmul(l1_calc, h_2_layer_weights), h_2_layer_bias))	
l3_calc = tf.nn.relu(tf.add(tf.matmul(l2_calc, h_3_layer_weights), h_3_layer_bias))
output_calc = tf.add(tf.matmul(l3_calc, output_weights), output_bias)

#defining training, cost, and accuracy calculations
cost = tf.reduce_mean(tf.square(Y - output_calc))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  #initalizes the varibles in prep for training


    for e in range(0, epochs):
        epoch_loss = 0 
        #s = np.arange(X_train.shape[0])
        shuffle_set = np.random.permutation(np.arange(len(Y_train))) #randomly shuffles data for each epoch to combat overfitting
        X_train = X_train[shuffle_set]
        Y_train = Y_train[shuffle_set]



        for i in range(0, len(Y_train) // batch_size):
            print("Batch: {0}".format(str(i) + "/" + str(len(Y_train) // batch_size)), end="\r")
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = Y_train[start:start + batch_size]
            _, c = sess.run([optimizer, cost], feed_dict ={X: batch_x, Y: batch_y}) #runs the batch in the optimizer and gets the cost
            epoch_loss += c #compounds cost for epoch

        if e % 1 == 0:
            testing_acc = cost.eval({X: X_val, Y: Y_val}) #evaluates accuracy of model on unseen data
            training_acc = cost.eval({X: X_train, Y: Y_train}) #evaluates accracy of model on seen data

            print('[+] -----   Epoch: ', e, 'Accuracy on Unseen Data:', testing_acc, 'Accuracy on TRAINING data: ', training_acc, 'Training Loss: ', epoch_loss)
    preds = sess.run([output_calc], feed_dict={X:X_val})

    for i in range(len(preds)):
        print(preds[i])
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(Y_val)