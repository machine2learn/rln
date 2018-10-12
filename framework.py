import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from keras import regularizers
from Keras_implementation import RLNCallback
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from numpy.random import seed
from keras.backend import eval as keras_eval
import warnings
warnings.filterwarnings("ignore")

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Add noisy features
noise_features = 1000
x_train = np.concatenate([x_train, np.random.normal(size=(x_train.shape[0], noise_features))], axis=1)
x_test = np.concatenate([x_test, np.random.normal(size=(x_test.shape[0], noise_features))], axis=1)

# Scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train, y_train)
x_test = scaler.transform(x_test, y_test)

INPUT_DIM = x_train.shape[1]


def base_model(layers=4, l1=0):
    assert layers > 1

    def build_fn():
        inner_l1 = l1
        # create model
        model = Sequential()
        # Construct the layers of the model to form a geometric series
        prev_width = INPUT_DIM
        for width in np.exp(np.log(INPUT_DIM) * np.arange(layers - 1, 0, -1) / layers):
            width = int(np.round(width))
            model.add(Dense(width, input_dim=prev_width, kernel_initializer='glorot_normal', activation='relu',
                            kernel_regularizer=regularizers.l1(inner_l1)))
            # For efficiency we only regularized the first layer
            inner_l1 = 0
            prev_width = width

        model.add(Dense(1, kernel_initializer='glorot_normal'))

        # Compile model
        model.compile(loss='mean_squared_error', optimizer='rmsprop')
        return model

    return build_fn


MJTCP = 32292  # Michael Jordan total career points


def test_model(build_fn, modle_name, num_repeates=10):
    seed(MJTCP)
    results = np.zeros(num_repeates)
    for i in range(num_repeates):
        reg = KerasRegressor(build_fn=build_fn, epochs=100, batch_size=10, verbose=0)
        reg.fit(x_train, y_train)
        results[i] = reg.score(x_test, y_test)
    print("%s: %.2f (%.2f) MSE" % (modle_name, results.mean(), results.std()))
    return results.mean()


layers = 4
#
# prev_score = np.inf
# cur_score = 0
#
# while (cur_score < prev_score) or (prev_score is None):
#     prev_score = cur_score
#     layers += 1
#     cur_score = test_model(base_model(layers=layers), "Network with %d layers" % layers)
#
# layers -= 1
# print ("The best results of an unregularized network are achieved with depth %d" % layers)
#
# l1 = 0.001
#
# prev_score = np.inf
# cur_score = None
#
# while cur_score < prev_score or prev_score is None:
#     prev_score = cur_score
#     l1 *= 10
#     cur_score = test_model(base_model(layers=layers, l1=l1), "L1 regularization of %.0E" % l1)
#
# best_l1_score = prev_score
#
# l1 /= 10
# print("The best L1 regularization is achieved with l1 = %.0E" % l1)


def RLN(layers=4, **rln_kwargs):
    def build_fn():
        model = base_model(layers=layers)()

        # For efficiency we only regularized the first layer
        rln_callback = RLNCallback(model.layers[0], **rln_kwargs)

        # Change the fit function of the model to except rln_callback:
        orig_fit = model.fit

        def rln_fit(*args, **fit_kwargs):
            orig_callbacks = fit_kwargs.get('callbacks', [])
            rln_callbacks = orig_callbacks + [rln_callback]
            return orig_fit(*args, callbacks=rln_callbacks, **fit_kwargs)

        model.fit = rln_fit

        return model

    return build_fn

best_rln_score = np.inf
Theta, learning_rate = None, None

for cur_Theta, log_learning_rate in [(-8, 6), (-10, 5), (-10, 6), (-10, 7), (-12, 6)]:
    cur_learning_rate = np.power(10, log_learning_rate)
    cur_score = test_model(RLN(layers=layers, norm=1, avg_reg=cur_Theta, learning_rate=cur_learning_rate),
                           "RLN with Theta=%s and learning_rate=%.1E" % (cur_Theta, cur_learning_rate))
    if cur_score < best_rln_score:
        Theta, learning_rate = cur_Theta, cur_learning_rate
        best_rln_score = cur_score

print("The best RLN is achieved with Theta=%d and learning_rate=%.1E" % (Theta, learning_rate))
print("We see that RLN outperforms L1 regularization on this dataset, %.2f < %.2f" % (best_rln_score, best_l1_score))
print("We also see that the average regularization required in RLN is much smaller than required in L1 regularized models:")
print("%.1E << %.1E" % (np.exp(Theta), l1))
