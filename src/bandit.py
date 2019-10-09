import numpy as np
import matplotlib.pyplot as plt

def Gaussian(mu,sigma):
    return np.random.normal(mu,sigma,1)

def simpleBandit(R,steps,eps):
    k = len(R)
    Q = []
    N = []
    for i in range(k):
        Q += [0]
        N += [0]
    idx_R_max = np.where(R == np.max(R))[0]
    reward = []
    optimalActions = []
    for i in range(steps):
        prob = np.random.uniform(0,1,1)[0]
        if prob >= eps:
            idx_Q_max = np.where(Q == np.max(Q))[0]
            a = int(np.random.uniform(idx_Q_max[0],idx_Q_max[-1]+1,1)[0])
        else:
            a = int(np.random.uniform(0,k,1)[0])
        Ri = Gaussian(R[a],1.0)[0]
        reward += [Ri]
        Q[a] = (Q[a]*N[a] + Ri)/(N[a]+1)
        N[a] += 1
        optimalActions += [int(a in idx_R_max)]

    return reward, optimalActions

def runSimpleBandit(k,steps,runs,eps):
    R = []
    for i in range(k):
        R += [Gaussian(0,1)[0]]
    print("R = ", R)
    data = [np.random.normal(R[i],1,steps) for i in range(k)]
    reward = []
    optAct = []
    ax1 = plt.subplot(1,2,1)
    ax1.violinplot(data,range(1,k+1),points=20,widths=0.8,showmeans=True,showextrema=False,showmedians=False)
    ax1.grid(axis='x')
    ax1.set_xticks(range(k+1))
    for e in range(len(eps)):
        avg_reward = [0 for i in range(steps)]
        percOptAct = [0 for i in range(steps)]
        for i in range(runs):
            # np.random.seed(i)
            reward, optimalActions = simpleBandit(R,steps,eps[e])
            avg_reward = [avg_reward[j] + reward[j]/runs for j in range(steps)]
            percOptAct = [percOptAct[j] + optimalActions[j]/runs for j in range(steps)]
        ax2 = plt.subplot(2,2,2)
        ax2.plot(range(steps),avg_reward)
        ax2.grid(True)
        ax3 = plt.subplot(2,2,4)
        ax3.plot(range(steps),percOptAct)
        ax3.grid(True)
    ax2.legend(eps)
    plt.show()

k = 10
steps = 1000
runs = 2000
eps = [0,0.01,0.1]
runSimpleBandit(k,steps,runs,eps)
