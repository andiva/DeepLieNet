import numpy as np
import matplotlib.pyplot as plt
import seaborn
from scipy.signal import savgol_filter, medfilt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error as MSE

import keras
from keras import backend as K
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, LSTM
import pickle

from Taylor_Map import TaylorMap

def create_PNN(N = 1, custom_loss=None, inputDim=2, outputDim=2, order=5):
    ''' Creates polynomial neural network based on Taylor map'''
    model = Sequential()
    tmap1 = TaylorMap(output_dim = outputDim, order=2,
                      input_shape = (inputDim,))
    tmap = TaylorMap(output_dim = outputDim, order=order,
                      input_shape = (inputDim,))
    tmap2 = TaylorMap(output_dim = outputDim, order=2,
                      input_shape = (inputDim,))

    model.add(tmap1)
    for i in range(N):
        model.add(tmap)
    model.add(tmap2)
    if custom_loss=='l1':
        rate = 0.0001
        def l1_reg(weight_matrix):
            return rate * K.sum(K.abs(weight_matrix))

        wts = model.layers[-1].trainable_weights # -1 for last dense layer.
        reg_loss = l1_reg(wts[0])
        for i in range(1, len(wts)):
            reg_loss += l1_reg(wts[i])

        def custom_loss(reg_loss):
            def orig_loss(y_true, y_pred):
                return K.mean(K.square(y_pred - y_true)) + reg_loss
            return orig_loss

        loss_function = custom_loss(reg_loss)
    else:
        loss_function='mean_squared_error'

    opt = keras.optimizers.Adamax(lr=0.02, beta_1=0.99,
                                   beta_2=0.99999, epsilon=1e-1, decay=0.0)
    model.compile(loss=loss_function, optimizer=opt)
    return model


def createLSTM(inputDim, outputDim):
    model = Sequential()
    model.add(LSTM(10, input_dim=inputDim, input_length=1))
    model.add(Dense(outputDim, init='uniform', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model


def createMLP(inputDim, outputDim):
    model = Sequential()
    model.add(Dense(4, input_dim=inputDim, init='uniform', activation='sigmoid'))
    model.add(Dense(4, init='uniform', activation='sigmoid'))
    model.add(Dense(outputDim, init='uniform', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adamax')
    return model


def predict(model, X0, epoch_n, reshape = False):
    ''' Predicts dynamics with PNN '''
    X = []
    X.append(X0)
    for i in range(epoch_n):
        x0 = X[-1]
        if reshape:
            x0 = x0.reshape(1,1,2)
        X.append(model.predict(x0))

    return np.array(X)


def smooth(x, window_len=11, window='hanning'):
    if window_len<3:
        return x

    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def load_data(id, window_len=11, dt=5):
    X = np.genfromtxt(str(id)+".csv", skip_header=0, delimiter=',')
    start_t = X[0,0]
    end_t = X[-1,0]
    t = np.arange(start_t, end_t+5, dt)
    X_interp = np.empty((len(t), X.shape[1]))
    X_interp[:,0] = t
    for j in range(1, X.shape[1]):
        X_interp[:, j] = np.interp(t, X[:, 0], X[:, j])

    for j in range(1, X.shape[1]):
        s1 = smooth(X_interp[:, j], window_len=window_len)
        X_interp[:, j] = s1[window_len//2:-window_len//2+1]
    X_interp[0,1] = X[0,1]
    return X_interp, X

def f(t, y):
    k = 0.933059439849773
    return np.array([0.01*y*(y-k)])

def predict2(X0, epoch_n):
    ''' Predicts dynamics with PNN '''
    X = []
    X.append(X0)
    k=1.000090419134
    for i in range(epoch_n):
        x0 = X[-1]
        #x0 = k*x0
        X.append(x0*0.954418550095225*k +
                 x0*x0*0.0466248766920578*k*k+
                 x0*x0*x0*0.00227769999475867*k*k*k)

    return np.array(X)


def fit(pnn, dt, X):
    N = int(X.shape[0]/dt)
    X0 = X[0, [1,2]].reshape((1,2))
    for k in range(10):
        for i in range(N-1):
            j = i*dt
            pnn.fit(X0, X[j+1, [1,2]].reshape((1,2)), epochs=10, verbose=1)
            X0 = pnn.predict(X0)


def get_weights(pnn):
    W  = pnn.layers[-1].get_weights()
    wr = W[0]
    for i in range(1, len(W)):
        wr = np.vstack((wr, W[i]))
    return wr.ravel()


def set_weights(pnn, wr):
    wr = wr.reshape((-1, 2))
    W=[wr[:1,:], wr[1:3,:], wr[3:7,:], wr[7:15,:], wr[15:31,:], wr[31:,:]]
    pnn.layers[-1].set_weights(W)


def get_trained_PNN():
    dt = 2
    A2 = 0.25
    a1 = 0.7277
    a2 = 0.6725
    a3 = 0.6270
    p = np.polyfit(np.array([600., 700., 800.]), np.array([a1, a2, a3]), deg=2)
    pc = np.polyfit(np.array([600., 700., 800.]), np.array([1./(a1-A2), 1./(a2-A2), 1./(a3-A2)]), deg=2)
    p2 = np.polyfit(np.array([1./(a1-A2), 1./(a2-A2), 1./(a3-A2)]), np.array([a1, a2, a3]), deg=2)
    p2c = np.polyfit(np.array([1./(a1-A2), 1./(a2-A2), 1./(a3-A2)]), np.array([600., 700., 800.]), deg=2)

    pnn = create_PNN(N=dt, order=5)

    W = pnn.layers[0].get_weights()
    W[0][0,0] = -p[2]
    W[1][1,0] = -p[1]
    W[2][3,0] = -p[0]

    W[0][0,1] = pc[2]
    W[1][1,1] = pc[1]
    W[2][3,1] = pc[0]
    pnn.layers[0].set_weights(W)
    # for w in W:
    #     print(w.T)
    # return

    W = pnn.layers[1].get_weights()
    W[1][0,0] = 1.04086747295101
    W[3][1,0] = 0.0425376232964091
    W[5][2,0] = 0.00173840516946467
    pnn.layers[1].set_weights(W)

    W = pnn.layers[2].get_weights()
    W[0][0,0] = p2[2]
    W[1][1,0] = p2[1]
    W[2][3,0] = p2[0]

    W[0][0,1] = p2c[2]
    W[1][1,1] = p2c[1]
    W[2][3,1] = p2c[0]
    pnn.layers[2].set_weights(W)
    return pnn


def main():
    dt = 2


    X1, X1_raw = load_data(600, dt=dt)
    X1[:,0]-=X1[0,0]
    X1_raw[:,0]-=X1_raw[0,0]
    X1[:, 2]/=-600
    plt.plot(X1_raw[:, 0], X1_raw[:, 1], label='data at 600mA')

    X2, X2_raw = load_data(700, dt=dt)
    X2[:,0]-=X2[0,0]
    X2_raw[:,0]-=X2_raw[0,0]
    X2[:, 2]/=-700
    plt.plot(X2_raw[:, 0], X2_raw[:, 1], label='data at 700mA')

    X3, X3_raw = load_data(800, dt=dt)
    X3[:,0]-=X3[0,0]
    X3_raw[:,0]-=X3_raw[0,0]
    X3[:, 2]/=-800
    plt.plot(X3_raw[:, 0], X3_raw[:, 1], label='data at 800mA')


    pnn = get_trained_PNN()
    #lstm = createMLP(2, 2)
    #X_train = np.vstack((X1[:, 1], 0.6*np.ones_like(X1[:, 1]))).T
    #num_epoch = 5000
    #xtr = X_train[:-1].reshape((-1, 1, X_train.shape[1]))
    #xtr = X_train[:-1]
    #lstm.fit(xtr, X_train[1:], nb_epoch=num_epoch, batch_size=50, verbose=1)
    #pnn = lstm


    N1=int((X1[-1,0]-X1[0,0])/dt)
    #Xn1 = predict(pnn, np.array([X1[0, 1] - a1, 1./(a1-A2)]).reshape((1,2)), N1)[:,0,:]
    Xn1 = predict(pnn, np.array([X1[0, 1], 600.]).reshape((1,2)), N1)[:,0,:]
    # return
    tn1 = np.arange(X1[0,0], X1[0,0]+N1*dt+1, dt)
    print(Xn1.shape)

    N2=int((X2[-1,0]-X2[0,0])/dt)
    # Xn2 = predict(pnn, np.array([X2[0, 1] - a2, 1./(a2-A2)]).reshape((1,2)), N2)[:,0,:]
    Xn2 = predict(pnn, np.array([X2[0, 1], 700]).reshape((1,2)), N2)[:,0,:]
    tn2 = np.arange(X2[0,0], X2[0,0]+N2*dt+1, dt)

    N3=int((X3[-1,0]-X3[0,0])/dt)
    # Xn3 = predict(pnn, np.array([X3[0, 1] - a3, 1./(a3-A2)]).reshape((1,2)), N3)[:,0,:]
    Xn3 = predict(pnn, np.array([X3[0, 1], 800]).reshape((1,2)), N3)[:,0,:]
    tn3 = np.arange(X3[0,0], X3[0,0]+N3*dt+1, dt)

    #Xn4 = predict(pnn, np.array([0.55, 900]).reshape((1,2)), N3)[:,0,:]



    plt.gca().set_prop_cycle(None)
    plt.plot(tn1, Xn1[:,0], 'o', markersize=3, fillstyle='none', linewidth=1, label='prediction at 600mA')
    plt.plot(tn2, Xn2[:,0], 'o', markersize=3, fillstyle='none', linewidth=1, label='prediction at 700mA')
    plt.plot(tn3, Xn3[:,0], 'o', markersize=3, fillstyle='none', linewidth=1, label='prediction at 800mA')
    #plt.plot(tn3, Xn4[:,0], 'o', markersize=3, fillstyle='none', linewidth=1, label='...')
    plt.grid()
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.show()
    return

    er = Xn[:,0]- X[:, 1]
    der = np.gradient(er)

    return 0

if __name__ == "__main__":
    main()