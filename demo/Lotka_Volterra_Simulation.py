import numpy as np
import sympy as sp
from sympy import Symbol, expand

import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import sys
sys.path.insert(0,'..')

from core.Lie import LieLayer
from core.Lie_map_builder import LieMapBuilder


def main():
    x = Symbol('x')
    y = Symbol('y')
    state = np.array([x,y])

    right_hand_side = [ -y-x*y, x+x*y]

    order = 3
    map_builder = LieMapBuilder(state, right_hand_side, order=order)
    for P in map_builder.P:
        print P

    R = map_builder.getInitR()
    R = map_builder.propogate(h=0.01, N=10)

    for Rk in R:
        print Rk

    W = map_builder.convert_weights_to_full_nn(R)
    dim = len(state)

    map = LieLayer(output_dim = dim, order = order, input_shape = (dim,))
    model = Sequential()
    model.add(map)
    map.set_weights(W)

    X0 = np.array([[0, 0.05],
                   [0, 0.15],
                   [0, 0.3]],
                  dtype=float).reshape((-1, dim))

    X = []
    X.append(X0)
    import time
    print('start simulation')
    start = time.time()
    for i in xrange(70):
        X.append(model.predict(X[-1])) 
    print('elapsed time: %s sec'% (time.time()-start))

    X = np.array(X)



    plt.plot(X[:,:,0], X[:,:,1], 'b.', markersize=3)
    plt.grid()
    
    
    
    # compare to ode solver
    from scipy.integrate import solve_ivp

    def model(t, state):
        x = state[0]
        y = state[1]
        return np.array([-y-x*y, x+x*y])
    
    #as soon as R = map_builder.propogate(h=0.01, N=10)
    # and calc for i in xrange(70):
    t = 0.01*10*70
    for state0 in X0:
        X = solve_ivp(model, [0, t], state0, max_step=0.1)
        plt.plot(X.y[0, :], X.y[1, :], 'r-', alpha = 0.4)
    
    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))