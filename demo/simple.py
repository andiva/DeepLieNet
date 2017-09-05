import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
import sys
sys.path.insert(0,'..')

from core.Lie import LieLayer

def createLieNet(inputDim=2, outputDim=2, order=2):
    model = Sequential()
    model.add(LieLayer(output_dim = outputDim, order=order,
                      input_shape = (inputDim,)))
    opt = keras.optimizers.Adamax(lr=0.02, beta_1=0.9,
                                  beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def main():
    # demo for y = 1.5 + x + x*x
    # generate data
    X = np.array(range(10)).reshape((-1,1))
    X = np.hstack((X, X+10))
    Y = 1.5 + X + X*X
    # split to train and test data sets 
    X_train = X[:, 0]
    Y_train = Y[:, 0]

    X_test = X[:, 1]
    Y_test = Y[:, 1]


    # create and fit Lie transform based neural network
    model = createLieNet(inputDim=1, outputDim=1, order=2)
    model.fit(X_train, Y_train, nb_epoch=10000, verbose=1)
    
    # draw results
    ax = plt.subplot(1,1,1)
    
    ax.plot(X_train, Y_train, 'b-', label='train')
    ax.plot(X_train, model.predict(X_train), 'r*')
    
    ax.plot(X_test, Y_test, 'g-', label='test')
    ax.plot(X_test, model.predict(X_test), 'r*', label='prediction')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)


    plt.show()



if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))