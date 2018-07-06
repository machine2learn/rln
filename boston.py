import pandas as pd
import numpy as np
np.random.seed(42)
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
tf.set_random_seed(42)
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from keras.datasets import boston_housing


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
noise_features = 1000
x_train = np.concatenate([x_train, np.random.normal(size=(x_train.shape[0], noise_features))], axis=1)
x_test = np.concatenate([x_test, np.random.normal(size=(x_test.shape[0], noise_features))], axis=1)

# Scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train, y_train)
x_test = scaler.transform(x_test, y_test)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

INPUT_DIM = x_train.shape[1]

BATCH_SIZE=10
EPOCHS = 800
learning_rate = 0.01

n_hidden_1 = 179 # 1st layer number of neurons
n_hidden_2 = 31 # 2nd layer number of neurons
n_hidden_3 = 5
n_input = 1000+13 # MNIST data input (img shape: 28*28)
n_classes = 1 # MNIST total classes (0-9 digits)
avg_reg = -10
rln_lr = np.power(10,6)

# x_train, y_train = balanced_subsample(x_train, y_train.reshape(-1))
# y_train = y_train.reshape(-1,1)


in_epoch = int(round(x_train.shape[0]/BATCH_SIZE))

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'b3': tf.Variable(tf.zeros([n_hidden_3])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

lambdas = {
    'h1': tf.Variable(avg_reg*tf.ones([n_input, n_hidden_1]))
}

rts =  {
    'h1': tf.Variable(tf.zeros([n_input, n_hidden_1]))
}


def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

def rln(prev_weights, weights, lambdas, rts, rln_lr, avg_reg, norm=1):

    assert norm in [1, 2], "Only supporting l1 and l2 norms at the moment"

    ops = []
    #avg_reg and lr per weight?

    for k in lambdas.keys():
        gradients = weights[k] - prev_weights[k]

        norms_derivative = weights[k]*2 if norm ==2 else tf.sign(weights[k])
        norms_derivative += np.finfo(np.float32).eps #TODO choose epsilon

        lambda_grad = tf.multiply(gradients, rts[k])

        lambda_update = lambdas[k] - rln_lr * lambda_grad

        translation = avg_reg - tf.reduce_mean(lambda_update)
        lambda_update +=  translation

        max_lambda_values = tf.log(tf.abs(weights[k]/norms_derivative)) 
        max_lambda_values = tf.where(tf.is_nan(max_lambda_values), tf.ones_like(max_lambda_values)*tf.reduce_max(lambdas[k]), max_lambda_values) #TODO upper/lower bounds

        lambda_update = tf.assign(lambdas[k], tf.clip_by_value(lambda_update, tf.reduce_min(lambdas[k]), max_lambda_values)) #TODO upper/lower bounds

        rts_update = tf.assign(rts[k], tf.multiply(norms_derivative,tf.exp(lambda_update)))

        weights_op = tf.assign(weights[k], weights[k] - rts_update)

        ops += [weights_op]


    return ops


# Construct model
logits = multilayer_perceptron(x)

prev_weights = {key: value+0 for (key,value) in weights.items()} #+0, little hack since tf.identity still returns a reference, could use tf.assign though

with tf.control_dependencies([v for v in prev_weights.values()]): #We need to ensure prev_weights is computed befor the optimization
    # Define loss and optimizer
    loss = tf.losses.mean_squared_error(y, logits)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

with tf.control_dependencies([train_op]):
    updates = [train_op] + rln(prev_weights, weights, lambdas, rts, rln_lr, avg_reg)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        train_loss = []
        train_acc = []
        idxs = np.arange(len(x_train))
        np.random.shuffle(idxs)
        idxs = np.array_split(idxs, in_epoch)
        for j in range(len(idxs)):
            sess.run(updates,feed_dict={x: x_train[idxs[j]], y: y_train[idxs[j]]})

            loss_value = sess.run([loss], feed_dict={x: x_train[idxs[j]], y: y_train[idxs[j]]})

            train_loss.append(loss_value)

        #print("Iter: {}, MSE: {:.4f}".format(i, np.mean(train_loss)))
        idxs = np.arange(len(x_test))
        idxs = np.array_split(idxs, int(round(x_test.shape[0]/BATCH_SIZE)))
        test_loss = []
        for j in range(len(idxs)):
            loss_value = sess.run([loss], feed_dict={x: x_test[idxs[j]], y: y_test[idxs[j]]})
            test_loss.append(loss_value)
      #  pred = np.round(pred)
        print("Iter: {}, Test MSE: {:.4f}".format(i, np.mean(test_loss)))






