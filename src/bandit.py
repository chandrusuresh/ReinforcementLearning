import numpy as np
import matplotlib.pyplot as plt

seed = np.random.seed(1)
def Gaussian(mu,sigma):
    return np.random.normal(mu,sigma,1)

def simpleBandit(k,steps):
    R = []
    Q = []
    N = []
    for i in range(k):
        R += [Gaussian(0,1)]
        Q += [0]
        N += [0]

    reward = []
    for i in range(steps):
        ## Sort the array Q, so max element is at the end
        Q_sorted = sorted(Q)
        idx_Q_sorted = np.argsort(Q)
        ## Get the index of the first element = max Q
        idx = Q_sorted.index(Q_sorted[-1])
        ## If index of first element = last then max value is non-unique
        if idx == len(Q_sorted)-1:
            a = idx_Q_sorted[idx]
        else:
            ind = int(np.random.uniform(idx,len(idx_Q_sorted)-1,1))
            # print(idx,len(idx_Q_sorted)-1,ind)
            a = idx_Q_sorted[ind]
        reward += [Q[a]]
        Ri = Gaussian(R[a],1.0)
        Q[a] = (Q[a]*N[a] + Ri)/(N[a]+1)
        N[a] += 1

    plt.plot(range(steps),reward)
    plt.show()

simpleBandit(10,1000)
