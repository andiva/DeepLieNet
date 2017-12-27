import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import sys
sys.path.insert(0,'..')

from core.Lie import LieLayer

def createLieNet(inputDim=2, outputDim=2, order=2):
    model = Sequential()
    model.add(LieLayer(output_dim = outputDim, order=order,
                      input_shape = (inputDim,)))
    opt = keras.optimizers.Adamax(lr=0.02, beta_1=0.99,
                                  beta_2=0.99999, epsilon=1e-1, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


def createMLP(inputDim, outputDim):
    model = Sequential()
    model.add(Dense(4, input_dim=inputDim, init='uniform', activation='sigmoid'))
    model.add(Dense(4, init='uniform', activation='sigmoid'))
    model.add(Dense(outputDim, init='uniform', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adamax')
    return model


def createLSTM(inputDim, outputDim):
    model = Sequential()
    model.add(LSTM(10, input_dim=inputDim, input_length=1))
    model.add(Dense(outputDim, init='uniform', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model


def f(t, X, a=-2, b=-1, d=-1, g=-1):
    x = X[0]
    y = X[1]
    return np.array([ a*x - b*x*y, d*x*y - g*y])


def calcSolution(X0, dt, N, a=-2, b=-1, d=-1, g=-1):
    ans = np.empty((N, 2))
    t = 0
    X = X0
    for i in xrange(N):
        k1 = f(t, X)
        k2 = f(t+dt/2.0, X+dt*k1/2.0)
        k3 = f(t+dt/2.0, X+dt*k2/2.0)
        k4 = f(t+dt, X + dt*k3)

        X = X + dt*(k1+2*k2+2*k3+k4)/6.0  
        ans[i] = X
        t += dt

    # normalization to the fixed point:
    ans -= np.array([g/d, a/b])
    return ans


def iterative_predict(model, X0, N, reshape = False):
    ans = np.empty((N, 2))
    X = X0.reshape(-1,2)
    for i in xrange(N):
        if reshape:
            X = model.predict(X.reshape(1,1,2))
        else:
            X = model.predict(X)
        ans[i] = X
    return np.vstack((X0, ans[:-1]))


def main():

    N = 465
    dt = 0.01

    X_train = calcSolution(np.array([1.5, 2.5]), dt, N)
    X_test_outer  = calcSolution(np.array([1.8, 2.8]), dt, N)
    X_test_inner  = calcSolution(np.array([1.1, 2.1]), dt, N)
    X_test_fixed  = calcSolution(np.array([1.0, 2.0]), dt, N)


    lie = createLieNet(2,2)
    mlp = createMLP(2,2)
    lstm = createLSTM(2,2)

    f, ax_all = plt.subplots(1,3)

    titles = ['Lie transform', 'MLP', 'LSTM']
    for i, model in enumerate([lie, mlp, lstm]):
        reshape = False
        xtrain = X_train[:-1]
        if model is lstm:
            reshape = True
            xtrain = xtrain.reshape((-1, 1, X_train.shape[1]))
        
        model.fit(xtrain, X_train[1:], nb_epoch=2000, batch_size=50, verbose=1)

        f, ax = plt.subplots(1,2)
        
        for a in [ax[0], ax_all[i]]:
            a.plot(X_train[:,0], X_train[:,1], 'b-', label='train data')
            a.plot(X_test_fixed[:,0], X_test_fixed[:,1], 'y*', label='test: fixed point')
            a.plot(X_test_outer[:,0], X_test_outer[:,1], 'c-', label='test: outer track')
            a.plot(X_test_inner[:,0], X_test_inner[:,1], 'g-', label='test: inner track')

        time = np.arange(N)*dt
        
        ax[1].plot(time, X_train[:,0], 'b-', label='train data')
        ax[1].plot(time, X_test_fixed[:,0], 'y-', label='test: fixed point')
        ax[1].plot(time, X_test_outer[:,0], 'c-', label='test: outer track')
        ax[1].plot(time, X_test_inner[:,0], 'g-', label='test: inner track')
        
        for k in [0,1]:
            handles, labels = ax[k].get_legend_handles_labels()
            ax[k].legend(handles, labels)
        
        for X in [X_train, X_test_outer, X_test_inner, X_test_fixed]:
            X_predict = iterative_predict(model, X[0], N, reshape = reshape)

            for a in [ax[0], ax_all[i]]:
                a.plot(X_predict[::15,0], X_predict[::15,1], 'r*')
                a.plot(X_predict[:,0], X_predict[:,1], 'r-', alpha=0.4)

            ax[1].plot(time[::15], X_predict[::15,0], 'r*')
            ax[1].plot(time, X_predict[:,0], 'r-', alpha=0.4)
        
        ax[0].set_title(titles[i]+': phase and time spaces')

    for i, title in enumerate(titles):
        ax_all[i].set_title(title)
    
    plt.show()



if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))