import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import sys
sys.path.insert(0,'..')

from core.Lie import LieLayer

def getPhaseStateVector(dim):
    state = []
    for i in xrange(dim):
        state.append(sp.Symbol('x%s'% str(i+1).zfill(2)))
    return sp.Array(state)


def load_Lie_map(state, filepath):
    print('load weights: ', filepath)
    state_power = np.array([1])
    order = 0

    weights = []
    W = []
    with open(filepath, 'r') as file:
        for row in file:
            row = row.replace('\r', '')
            row = row.replace('\n', '')

            if(0==len(row)):
                W = np.array(W)
                W_ext = np.zeros((len(state), len(state_power)))
                str_power = state_power.astype(str)
                reduced_str_power = []
                sl = []
                for i, el in enumerate(str_power):
                    if el not in reduced_str_power:
                        reduced_str_power.append(el)
                        sl.append(i)
                
                W_ext[:, sl] = W
                weights.append(W_ext.T)
                state_power = np.kron(state_power, state)
                W = []
                continue

            if row[0] != '%':
                row = map(float, row.split(' '))
                W.append(row)
    return weights


def main():
    dim = 10  # predefined state vector dimension
    order = 3 # predefined order of nonlinearity

    DH = LieLayer(output_dim = dim, order = order, input_shape = (dim,))
    QFA= LieLayer(output_dim = dim, order = order, input_shape = (dim,))
    QDA= LieLayer(output_dim = dim, order = order, input_shape = (dim,))
    OQS= LieLayer(output_dim = dim, order = order, input_shape = (dim,))
    
    fodo = Sequential()
    lattice = [QDA, OQS, DH, OQS, QFA, OQS, DH, OQS, QDA, 
               QDA, OQS, DH, OQS, QFA, OQS, DH, OQS, QDA, 
               QDA, OQS, DH, OQS, QFA, OQS, DH, OQS, QDA,
               QDA, OQS, DH, OQS, QFA, OQS, DH, OQS, QDA]
    for el in lattice:
        fodo.add(el)

    state = getPhaseStateVector(dim=dim)
    DH_weights = load_Lie_map(state, filepath='data/accelerator/DH.txt')
    QFA_weights = load_Lie_map(state, filepath='data/accelerator/QFA.txt')
    QDA_weights = load_Lie_map(state, filepath='data/accelerator/QDA.txt')
    OQS_weights = load_Lie_map(state, filepath='data/accelerator/OQS.txt')

    DH.set_weights(DH_weights)
    QFA.set_weights(QFA_weights)
    QDA.set_weights(QDA_weights)
    OQS.set_weights(OQS_weights)

    
    X0 = np.array([0.001,0.001,0,0,0,0,0,0,1,0]).reshape((-1, 10))
    X0 = np.array([0.001,0.000,0,0,0,0,0,0,1,0]).reshape((-1, 10))
    X = []
    X.append(X0)
    for i in xrange(1000):
        X.append(fodo.predict(X[-1])) 
    
    X = np.array(X)


    print(X.shape)
    plt.plot(X[:, 0, 0], X[:, 0, 3], 'b.', markersize = 2)
    # plt.plot(X[:, 0, -1], X[:, 0, 8], 'b-', markersize = 2)
    plt.show()

    return


if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))