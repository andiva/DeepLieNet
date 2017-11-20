import numpy as np
import adaptfilt as adf

import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import sys
sys.path.insert(0,'..')

from core.Lie import LieLayer


def createLieNet(inputDim=2, outputDim=2, order=2):
    """Creates polynomial neural network"""
    model = Sequential()
    model.add(LieLayer(output_dim = outputDim, order=order,
                      input_shape = (inputDim,)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def nlms(u, d, M = 100):
    """A classical adaptive filtering"""
    # Apply adaptive filter
    M = 100  # Number of filter taps in adaptive filter
    step = 0.1  # Step size
    y, e, w = adf.nlms(u, d, M, step, returnCoeffs=True)

    return y, e, w


def neural_filter(u, d, epoch = 1, M = 100):
    window = M
    u_rolling = rolling_window(u, window)
    d_output = d[window-1:]
    d_output = d_output[:u_rolling.shape[0]].reshape((-1,1))
    
    # generate a polynomial neural network
    # order is order of nonlinearity
    nnet = createLieNet(u_rolling.shape[1], 1, order=1)
    # fit neural net
    nnet.fit(u_rolling, d_output, nb_epoch=epoch, verbose=1)
    d = d_output.ravel()
    y = nnet.predict(u_rolling).ravel()
    e = d - y

    return y, e, nnet.get_weights()


def main():
    u = np.load('data/speech.npy')

    coeffs = np.concatenate(([0.8], np.zeros(8), [-0.7], np.zeros(9),
                         [0.5], np.zeros(11), [-0.3], np.zeros(3),
                         [0.1], np.zeros(20), [-0.05]))    

    d = np.convolve(u, coeffs)

    # Add background noise
    v = np.random.randn(len(d)) * np.sqrt(5000)
    d += v

    # process adaptive filter
    y_ad, e_ad, w_ad = nlms(u, d)
    
    # process neural network fitting (increase number of epochs for higher precision) 
    y_nn, e_nn, w_nn, net = neural_filter(u, d, epoch = 1)


    plt.plot(e_nn, label='Neural network filter error')
    plt.plot(e_ad, label='Adaptive filter error')
    plt.legend()

    plt.figure()
    plt.plot(w_ad[-1], 'g', label='Estimated coefficients adaptive')
    plt.plot(coeffs, 'b--', label='Real coefficients')
    plt.legend()

    plt.figure()
    plt.plot(w_nn[-1], 'g', label='Estimated coefficients neural')
    plt.plot(coeffs, 'b--', label='Real coefficients')
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))