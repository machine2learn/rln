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

    def set_prediction(self):
        for d in self.dense:
            d.training = False

    def set_training(self):
        for d in self.dense:
            d.training = True


def loss(model, inputs, targets):
    model.set_prediction()
    result = tf.losses.mean_squared_error(targets, model(inputs))
    model.set_training()
    return result


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)

    return tape.gradient(loss_value, model.variables)


class RLN(tf.keras.layers.Layer):
    def __init__(self, output_units):
        super().__init__()
        self.output_units = output_units
        self.theta = -12
        self.norm = 2
        self.etha = 0.01
        self.mu = 10e6

    def build(self, input_shape):
        self.kernel = self.add_variable(
            "kernel", [int(input_shape[-1]), self.output_units])

        self.bias = self.add_variable('bias', [self.output_units])

        lambdas_initializer = tf.constant_initializer(self.theta * np.ones([int(input_shape[-1]), self.output_units]))

        self.lambdas = self.add_weight(name='lambdas',
                                       shape=([int(input_shape[-1]), self.output_units]),
                                       initializer=lambdas_initializer,
                                       trainable=False)

        self.rs = self.add_weight(name='rs',
                                  shape=([self.output_units]),
                                  initializer='zeros',
                                  trainable=False)

        self.prev = self.add_weight(name='prev_kernel',
                                    shape=[int(input_shape[-1]), self.output_units],
                                    initializer='zeros',
                                    trainable=False)

        self.first_time = False

        self.training = True

    def call(self, input, **kwargs):
        if self.training:
            if self.first_time:
                g = self.kernel - self.prev

                self.lambdas = self.lambdas + self.mu * self.etha * self.rs

                norms_derivative = self.kernel * 2 if self.norm == 2 else tf.sign(self.kernel)
                norms_derivative += np.finfo(np.float32).eps

                projected = self.theta - tf.reduce_mean(self.lambdas)
                self.lambdas = tf.add(self.lambdas, projected)

                max_lambda_values = tf.math.log(tf.abs(self.kernel / norms_derivative))
                self.lambdas = tf.clip_by_norm(self.lambdas, max_lambda_values)

                self.rs = tf.math.exp(self.lambdas) * norms_derivative

                self.kernel = self.prev - self.etha * (g + self.rs)  # prev not OK

            self.first_time = True
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

    for i in range(1000):
        grads = grad(model, training_inputs, training_outputs)
        # prev = [tf.identity(i) for i in model.variables]
        # prev = model.get_weights()
        optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())
        # model.set_weights(prev)
        # model.
        if i % 20 == 0:
            print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

    print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

    model.set_prediction()

    print(model([[12.0], [14.0], [5.0]], training=False))
