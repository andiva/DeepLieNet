import numpy as np
import sympy as sp
from sympy import Symbol, expand
import pickle

import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import sys
sys.path.insert(0,'..')

from core.Lie import LieLayer
from core.Lie_map_builder import LieMapBuilder

from scipy.integrate import solve_ivp

def model(t, state):
    a1 = 0.238 
    b1 = -0.260
    m1 = 1798.18*0.1

    p2 = 0.011 
    a2 = 0.172
    m2 = 10


    z1 = state[0]
    z2 = state[1]

    dstate = np.array([((a1*z1 + b1*z2)/(m1 + m2))*(m1 - z1 + m2 - z2),
                       (p2 + a2*z2/m2)*(m2 - z2)])
    print dstate
    return dstate


def model_full(t, state):
    p1 = -0.010
    a1 =  0.526 
    b1 = -0.840
    al2=  0.998
    m1 = 1347.570

    p2 = 0.011 
    a2 = 0.167
    b2 = 1.058
    al1= 0.001
    m2 = 378.76

    z1 = state[0]
    z2 = state[1]

    return np.array([(p1+(a1*z1 + al2*b1*z2)/(m1 + al2*m2))*(m1 - z1 + al2*(m2 - z2)),
                     (p2+(a2*z2 + al1*b2*z1)/(m2 + al1*m1))*(m2 - z2 + al1*(m1 - z1))])


def iterative_predict(model, X0, N, st_size=4):
    ans = np.empty((N, st_size))
    X = X0.reshape(-1,st_size)
    for i in xrange(N):
        X = model.predict(X)
        ans[i] = X
    return np.vstack((X0, ans[:-1]))

def moving_average(x, window_len=4):
        w = np.ones(window_len)
        s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        return np.convolve(w/w.sum(), s, mode='valid')


def main():
    # X0 = np.array([8, 0])
    # X = solve_ivp(model, [13, 100], X0, max_step=0.1)
    
    # plt.plot(X.t, X.y[0, :], 'r-', alpha = 0.4)
    # plt.plot(X.t, X.y[1, :], 'r-', alpha = 0.4)
    # plt.grid()
    # plt.show()

    iPhone = np.array([0, 2, 3, 2, 1, 7, 5, 4, 6, 7, 9, 8, 9,14,16,19,20,17,36,35,25,26,47,36,30,34,50,44,35,39,74,61,46,47,75,50], dtype=float)
    iPad =   np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 7, 5, 9,11,15,12,17,14,23,20,15,14,25,16,13,12,21,13,11,10,16,10], dtype=float)

    

    plt.plot(iPhone, 'bo-', markersize=3, linewidth=0.5, alpha=0.5)
    plt.plot(iPad, 'ro-', markersize=3, linewidth=0.5, alpha=0.5)

    n=len(iPhone)

    # for i in xrange(6):
    iPhone=moving_average(iPhone, window_len=5)[:n]
    iPad=moving_average(iPad, window_len=5)[:n]

    # plt.plot(iPhone, 'bo-', markersize=3, linewidth=0.5, alpha=0.3)
    # plt.plot(iPad, 'ro-', markersize=3, linewidth=0.5, alpha=0.3)

    
    # plt.show()
    # quit()
    
    # plt.xlim(-1, 41)
    # plt.ylim(-2, 82)

    st_size = 2
    inputDim = st_size
    outputDim = st_size
    order = 5
    model = Sequential()
    map = LieLayer(output_dim = outputDim, order=order,
                   input_shape = (inputDim,))
    model.add(map)
    opt = keras.optimizers.Adamax(lr=0.02, beta_1=0.99,
                                  beta_2=0.2, epsilon=1e-1, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='adamax')
    
    

    scale = 100
    data = np.vstack((iPhone, iPad)).T

    fix_point = np.array([65, 9]) 
    data-=fix_point

    data/=scale

    avg = np.average(data, axis=0)
    # data-=avg

    # data = np.hstack((data,np.vstack((np.gradient(data[:,0]), np.gradient(data[:,1]))).T))

    start_dynamics = 12
    N = len(data)


    import os
    filename = 'apple.2.pkl'
    filename2 = 'apple.3.pkl'
    if os.path.isfile(filename):
        with open(filename, 'rb') as file:
            R = pickle.load(file)
        map.set_weights(R)

    # model.fit(data[start_dynamics:-1], data[start_dynamics+1:], nb_epoch=6000, batch_size=20, verbose=1)
    # with open(filename2, 'wb') as file:
    #     pickle.dump(map.get_weights(), file)


    N-=10
    X_predict = iterative_predict(model, data[start_dynamics], N, st_size=st_size)
    X_predict = X_predict[:,:2]
    # X_predict+=avg
    X_predict*=scale
    X_predict+=fix_point

    print(X_predict[-1,:])

    # N-=12
    plt.plot(range(start_dynamics, start_dynamics+N), X_predict[:,0], 'm-', alpha=1)
    plt.plot(range(start_dynamics, start_dynamics+N), X_predict[:,1], 'y-', alpha=1)


    plt.grid()
    plt.show()

    

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))    
    