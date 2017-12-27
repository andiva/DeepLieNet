import numpy as np
import sympy as sp
from sympy import Symbol, expand


def getKronPowers(state, order, dim_reduction = True):
    """Calculates Kroneker powers of state vector
       with dimension reduction

    e.g. for (x y) and order=2 returns:
    1, (x y), (x x*y y)  

    Returns:
    list of numpy arrays, index corresponds to power 
    """
    powers = [state]
    index = [np.ones(len(state), dtype=bool)]
    for i in xrange(order-1):
        state_i = np.kron(powers[-1], state)
        reduced, red_ind = reduce(state_i)
        if dim_reduction:
            powers.append(reduced)
        else:
            powers.append(state_i)
            
        index.append(red_ind)

    powers.insert(0, np.array([1]))
    index.insert(0, [True])
    return powers, index 

def reduce(state):
    state_str = state.astype(str)
    reduced_state = []
    unique = []

    index = []
    for variable, variable_str in zip(state, state_str):
        if variable_str not in unique:
            unique.append(variable_str)
            index.append(True)
            reduced_state.append(variable)
        else:
            index.append(False)

    return np.array(reduced_state), index


def fill_matrices(state_powers, expr, M):
    coefs = [e.as_coefficients_dict() for e in expr]

    for k, state_power in enumerate(state_powers):
        for i, coef_dict in enumerate(coefs):
            for j, variable in enumerate(state_power):
                M[k][i, j] += coef_dict[variable]
    return M


def sum(X, Y, k):
    res = []
    for x, y in zip(X,Y):
        res.append(x+k*y)
    return res


def printlist(l):
    for el in l:
        print el
    return

class LieMapBuilder:
    def __init__(self, state, right_hand_side, order=3):
        """Initialization of Lie map builder

        Arguments:
        state -- 1d numpy array of sympy Symbols
        right_hand_side -- iterable  sympy expressions (polynomial)
        order -- order of nonlinearity of the resulting map 

        TO DO:
        authomatic expand right_hand_side into polynomial series
        """
        if len(state)!=len(right_hand_side):
            raise Exception('incorrect system dimension')

        self.StateSize = len(state)
        if order < 1:
            order = 1 # linear system at least
        print(state)
        print(right_hand_side)

        self.Order = order
        self.X, self.index = getKronPowers(state, order=order) 
        _,self.index_full = getKronPowers(state, order=order, dim_reduction=False)     
        
        self.P = []
        for X in self.X:
            self.P.append(np.zeros((self.StateSize, len(X))))

        fill_matrices(self.X, right_hand_side, self.P)
        
        return
    
    def right_hand_side_maps(self, R, verbose=False):
        dR = []
        n = self.StateSize
        for X in self.X:
            dR.append(np.zeros((n, len(X))))
        
        dR[0]+=self.P[0]


        X = np.zeros((self.StateSize,), dtype=object)
        for R_k, X0_k in zip(R, self.X):
            X += np.dot(R_k,X0_k)
        
        if verbose:
            print '>'
            print X
            print '---'
            


        Xk = np.array([1])

        for k in xrange(1, self.Order+1):
            Xk = np.kron(Xk, X)[self.index[k]]
            for i, xk in enumerate(Xk):
                Xk[i] = expand(xk)
            if verbose:
                print '>'
                print Xk
                print '---'
                # quit()
            fill_matrices(self.X, np.dot(self.P[k], Xk), dR)
        
        if verbose:
            quit()

        return dR

    def getInitR(self):
        R = []
        for X in self.X:
            R.append(np.zeros((self.StateSize, len(X))))
        
        R[1] = np.eye(self.StateSize)
        return R

    
    def propogate(self, h = 0.01, N = 10, R=None):
        if R == None:
            R = self.getInitR()
        
        for i in xrange(N):
            print i, N
            k1 = self.right_hand_side_maps(R)
            k2 = self.right_hand_side_maps(sum(R, k1, h/2), verbose=False)

            k3 = self.right_hand_side_maps(sum(R, k2, h/2))
            k4 = self.right_hand_side_maps(sum(R, k3, h/2))

            k12 = sum(k1, k2, 2)
            k34 = sum(k4, k3, 2)
            k = sum(k12, k34, 1)
            
            R = sum(R, k, h/6)
        return R

    def convert_weights_to_full_nn(self, R):
        W = []

        m = 1

        for ind, Rk in zip(self.index_full, R):
            w = np.zeros((self.StateSize, m))
            m*=self.StateSize
            w[:, ind] = Rk
            W.append(w.T)

        return W


def main():
    x = Symbol('x')
    y = Symbol('y')
    state = np.array([x,y])

    a=-2; b=-1; d=-1; g=-1

    # right_hand_side = [ a*x - b*x*y, d*x*y - g*y]
    right_hand_side = [ -y-x*y, x+x*y]

    map_builder = LieMapBuilder(state, right_hand_side, order=3)
    printlist(map_builder.P)

    R = map_builder.getInitR()
    print '---------------------'
    # dR = map_builder.right_hand_side_maps(R)
    # printlist(dR)

    R = map_builder.propogate(h=0.001, N=10)
    # printlist(R)
    # quit()

    x = [np.array([0, 0.3])]

    for i in xrange(100):
        x0 = x[-1]
        x0_2 = np.array([x0[0]*x0[0], x0[0]*x0[1], x0[1]*x0[1]])

        x.append(np.dot(R[1], x0)+np.dot(R[2], x0_2))
    
    x=np.array(x)
    import matplotlib.pyplot as plt

    plt.plot(x[:,0], x[:,1], 'b.', markersize=1)
    plt.show()

    return

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))