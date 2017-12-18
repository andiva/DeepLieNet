import numpy as np
import matplotlib.pyplot as plt
import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import sys
sys.path.insert(0,'..')

from core.Lie import LieLayer
from core.Lie_map_builder import LieMapBuilder

from sympy import Symbol
def main():
    m1 = Symbol('m01')
    m2 = Symbol('m02')
    m3 = Symbol('m03')
    m4 = Symbol('m04')
    m5 = Symbol('m05')
    m6 = Symbol('m06')
    m7 = Symbol('m07')
    m8 = Symbol('m08')
    m9 = Symbol('m09')
    m10 = Symbol('m10')
    m11 = Symbol('m11')

    k1 = 0.53
    k2 = 0.0072
    k3 = 0.625
    k4 = 0.00245
    k5 = 0.0315
    k6 = 0.6
    k7 = 0.0075
    k8 = 0.071
    k9 = 0.92
    k10 = 0.00122
    k11 = 0.87

    r1 = k1*m1*m2
    r2 = k2*m3
    r3 = k3*m3*m9
    r4 = k4*m4
    r5 = k5*m4
    r6 = k6*m5*m7
    r7 = k7*m8
    r8 = k8*m8
    r9 = k9*m6*m10
    r10 = k10*m11
    r11 = k11*m11

    right_hand_side = [
        r2+r5-r1,
        r2+r11-r1,
        r1+r4-r2-r3,
        r3-r4-r5,
        r5+r7-r6,
        r5+r10-r9,
        r7+r8-r6,
        r6-r7-r8,
        r4+r8-r3,
        r10+r11-r9,
        r9-r10-r11]
    
    state = np.array([m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11])

    map_builder = LieMapBuilder(state, right_hand_side, order=2)
    # R = map_builder.getInitR()
    # R = map_builder.propogate(h=0.002, N=5)

    # with open('map.pkl', 'wb') as file:
    #     pickle.dump(R, file)

    with open('map.pkl', 'rb') as file:
        R = pickle.load(file)

    W = map_builder.convert_weights_to_full_nn(R)
    dim = len(state)

    map = LieLayer(output_dim = dim, order = 2, input_shape = (dim,))
    model = Sequential()
    model.add(map)

    map.set_weights(W)

    X0 = np.array([[1,1,0,0,0,0,1,0,1,1,0],
                   [1,1,0,0,1,0,1,0,0,1,0]],
                  dtype=float).reshape((-1, 11))

    X = []
    X.append(X0)
    import time
    print('start simulation')
    start = time.time()
    for i in xrange(10000):
        X.append(model.predict(X[-1])) 
    print('elapsed time: %s sec'% (time.time()-start))

    X = np.array(X)
    print X.shape

    fig, ax = plt.subplots(1, 2)
    for i in xrange(2):
        ax[i].plot(np.arange(0, X.shape[0]), X[:, i, :], markersize = 1)
        ax[i].grid()
    plt.show()

    return

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))

