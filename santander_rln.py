import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

tf.set_random_seed(42)
np.random.seed(42)

BATCH_SIZE = 128
EPOCHS = 400
learning_rate = 0.001
df = pd.read_csv('train.csv').astype(np.float32)
labels = df['TARGET'].values
del df['TARGET']
del df['ID']
data = df.values
n_hidden_1 = 51  # 1st layer number of neurons
n_hidden_2 = 7  # 2nd layer number of neurons
n_input = 369  # MNIST data input (img shape: 28*28)
n_classes = 1  # MNIST total classes (0-9 digits)
avg_reg = -12
rln_lr = np.power(10, 6)

data = normalize(data, axis=0)

labels = labels.reshape(-1, 1).astype(np.float32)
x_train, x_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=0.3, random_state=5,
                                                                    stratify=labels)

# x_train, y_train = balanced_subsample(x_train, y_train.reshape(-1))
# y_train = y_train.reshape(-1,1)


in_epoch = int(round(x_train.shape[0] / BATCH_SIZE))

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

lambdas = {
    'h1': tf.Variable(avg_reg * tf.ones([n_input, n_hidden_1])),
    'h2': tf.Variable(avg_reg * tf.ones([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(avg_reg * tf.ones([n_hidden_2, n_classes]))
}

rts = {
    'h1': tf.Variable(tf.zeros([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.zeros([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.zeros([n_hidden_2, n_classes]))
}


def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


def rln(prev_weights, weights, lambdas, rts, rln_lr, avg_reg, norm=2):
    assert norm in [1, 2], "Only supporting l1 and l2 norms at the moment"

    ops = []
    # avg_reg and lr per weight?

    for k in lambdas.keys():
        gradients = weights[k] - prev_weights[k]

        norms_derivative = weights[k] * 2 if norm == 2 else tf.sign(weights[k])
        norms_derivative += np.finfo(np.float32).eps  # TODO choose epsilon

        lambda_grad = tf.multiply(gradients, rts[k])

        lambda_update = lambdas[k] - rln_lr * lambda_grad

        translation = avg_reg - tf.reduce_mean(lambda_update)
        lambda_update += translation

        max_lambda_values = tf.log(tf.abs(weights[k] / norms_derivative))
        max_lambda_values = tf.where(tf.is_nan(max_lambda_values),
                                     tf.ones_like(max_lambda_values) * tf.reduce_max(lambdas[k]),
                                     max_lambda_values)  # TODO upper/lower bounds

        lambda_update = tf.assign(lambdas[k], tf.clip_by_value(lambda_update, tf.reduce_min(lambdas[k]),
                                                               max_lambda_values))  # TODO upper/lower bounds

        rts_update = tf.assign(rts[k], tf.multiply(norms_derivative, tf.exp(lambda_update)))

        weights_op = tf.assign(weights[k], weights[k] - rts_update)

        ops += [weights_op]

    return ops


# Construct model
logits = multilayer_perceptron(x)

prev_weights = {key: value + 0 for (key, value) in weights.items()}
# +0, little hack since tf.identity still returns a reference, could use tf.assign though

with tf.control_dependencies(
        prev_weights.values()):  # We need to ensure prev_weights is computed befor the optimization
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

predicted = tf.nn.sigmoid(logits)
correct_pred = tf.equal(tf.round(predicted), y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
        for j in idxs:
            sess.run(updates, feed_dict={x: x_train[j], y: y_train[j]})

            loss_value, accuracy_value = sess.run([loss, accuracy], feed_dict={x: x_train[j], y: y_train[j]})

            train_loss.append(loss_value)
            train_acc.append(accuracy_value)

        print("Iter: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(i, np.mean(train_loss), np.mean(train_acc)))
        idxs = np.arange(len(x_test))
        idxs = np.array_split(idxs, int(round(x_test.shape[0] / BATCH_SIZE)))
        pred = []
        for j in idxs:
            _, _, p = sess.run([loss, accuracy, predicted],
                                                     feed_dict={x: x_test[j], y: y_test[j]})
            pred.append(p.squeeze())
        pred = np.concatenate(pred, axis=0)
        #  pred = np.round(pred)
        print("Iter: {}, AUC: {:.4f}".format(i, roc_auc_score(y_test.reshape(-1), pred)))
