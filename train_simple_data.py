import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
import csv

RANDOM_SEED = 44
tf.set_random_seed(RANDOM_SEED)

def init_weights(shape, num):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights, name=str('w'+str(num)))

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def get_simple_agent_data():
    """ Read the data set and split them into training and test sets """
    jdata = json.load(open('simple_data2.json'))
    print('opened file')
    jvals = len(jdata)
    print(jvals)
    #print(jdata[2]['state'][2])
    flen = len(jdata[0]['state'])
    data = np.zeros((jvals,flen))
    target = np.zeros((jvals,))
    for d in range(jvals):
        for e in range(flen):
            data[d][e] = jdata[d]['state'][e]
            target[d] = jdata[d]['actions'][0]

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    num_labels = 6 #len(np.unique(target))
    #print(num_labels)
    #all_Y = np.eye(jvals)[target.astype(int)]  # One liner trick! indexes must be ints though
    all_Y = np.zeros((jvals,num_labels))
    #print(all_Y.shape)
    for row in range(jvals):
        all_Y[row][int(target[row])] = 1 
    m,n,o,p = train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)
    print(len(m), len(n), len(o), len(p))
    #print(m[0])
    return m,n,o,p

    #train_test_index = int(jvals*0.8)
    #return all_X[0:train_test_index,:], all_X[train_test_index+1:jvals,:], #target[0:train_test_index,0], target[train_test_index+1:jvals,0] 

def main():
    train_X, test_X, train_y, test_y = get_simple_agent_data()
    #print(train_X.shape)
    #print(train_y.shape)
    #print(test_X.shape)
    #print(test_y.shape)
    #print(train_y[0])
    # Layer's sizes
    training_vals = train_X.shape[0]
    testing_vals = test_X.shape[0]
    x_size = train_X.shape[1]   # Number of input nodes: 2374 features and 1 bias
    h_size = 256                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of output nodes: 6 actions
    print(x_size, h_size, y_size)

    # Symbols
    X = tf.placeholder("float", shape=[None,x_size])
    y = tf.placeholder("float", shape=[None,y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size), 1)
    w_2 = init_weights((h_size, y_size), 2)

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    learning_rate = 0.01
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Run SGD
    saver = tf.train.Saver()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(100):
        # Train with each example
        for i in range(training_vals):
            #print(epoch, i, training_vals)
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) == sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) == sess.run(predict, feed_dict={X: test_X, y: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

        f = open('epochs_my_model_1.csv', 'a+')
        f.write(str(epoch)+','+str(train_accuracy)+','+str(test_accuracy)+'\n')
        f.close()

    # Save model for loading later
    saver.save(sess, './my_model_1')
    #sess.close()


if __name__ == '__main__':
    main()
