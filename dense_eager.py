import tensorflow as tf
import numpy as np
from functools import reduce

tf.enable_eager_execution()

tf.set_random_seed(42)
np.random.seed(42)


class Model(tf.keras.Model):

    def __init__(self, hidden_units):
        super().__init__()

        # self.dense = [tf.keras.layers.Dense(units=units) for units in hidden_units]
        self.dense = [RLN(units) for units in hidden_units]

    def call(self, input, **kwargs):
        result = reduce(lambda acc, layer: layer(acc), self.dense, input)
        return result


def loss(model, inputs, targets):
    return tf.losses.mean_squared_error(targets, model(inputs))


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)

    return tape.gradient(loss_value, model.variables)


class RLN(tf.keras.layers.Layer):
    def __init__(self, output_units):
        super().__init__()
        self.output_units = output_units
        self.thetat = 0.7
        self.norm = 2

    def build(self, input_shape):
        self.kernel = self.add_variable(
            "kernel", [int(input_shape[-1]), self.output_units])

        self.bias = self.add_variable('bias', [self.output_units])

        self.lambdas = self.add_weight(name='lambdas',
                                       shape=(self.output_units),
                                       initializer='he_normal',
                                       trainable=False)

        self.rs = self.add_weight(name='rs',
                                  shape=(self.output_units),
                                  initializer='zeros',
                                  trainable=False)

        self.prev = self.add_variable(
            "prev_kernel", [int(input_shape[-1]), self.output_units])

    def call(self, input, **kwargs):
        g = self.kernel - self.prev

        norms_derivative = self.kernel * 2 if self.norm == 2 else tf.sign(self.kernel)
        norms_derivative += np.finfo(np.float32).eps

        projected = self.theta - tf.reduce_sum(self.lambdas)
        self.lambdas = tf.add(self.lambdas, projected)
        self.rs = tf.math.exp(self.lambdas) * norms_derivative

        self.kernel = self.prev - self.etha(g + self.rs)  # prev not OK

        self.prev = tf.identity(self.kernel)
        return tf.matmul(input, self.kernel) + self.bias


# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
training_inputs = tf.random_normal([NUM_EXAMPLES, 1])
noise = tf.random_normal([NUM_EXAMPLES, 1])
training_outputs = training_inputs * 3 + 2
# training_outputs = training_inputs * 3 + 2 + noise

if __name__ == '__main__':
    model = Model([10, 5, 1])
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    for i in range(300):
        grads = grad(model, training_inputs, training_outputs)
        # prev = [tf.identity(i) for i in model.variables]
        # prev = model.get_weights()
        optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())
        # model.set_weights(prev)
        # model.
        if i % 20 == 0:
            print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

    print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
    print(model([[12.0], [14.0], [5.0]]))
