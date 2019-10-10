import numpy as np

def Gaussian(mu,sigma):
    return np.random.normal(mu,sigma,1)

def simpleBandit(R,steps,eps,init = 0,alpha=0,randomWalkSigma = 0):
    k = len(R)
    Q = []
    N = []
    for i in range(k):
        Q += [init]
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
        N[a] += 1
        if alpha == 0:
            Q[a] += (Ri - Q[a])/N[a]
        else:
            Q[a] += alpha*(Ri - Q[a])
        if randomWalkSigma != 0:
            Q = [Q[s] + Gaussian(0,randomWalkSigma)[0] for s in range(k)]
        optimalActions += [a]
    return reward, optimalActions

def runSimpleBandit(R,steps,runs,eps,init=0,alpha=0,randomWalkSigma = 0):
    rewards = []
    optAct = []
    idx_R_max = np.where(R == np.max(R))[0]
    for e in range(len(eps)):
        avg_reward = [0 for i in range(steps)]
        percOptAct = [0 for i in range(steps)]
        for i in range(runs):
            reward, optimalActions = simpleBandit(R,steps,eps[e],init,alpha,randomWalkSigma)
            avg_reward = [avg_reward[j] + reward[j]/float(runs) for j in range(steps)]
            percOptAct = [percOptAct[j] + float(optimalActions[j] in idx_R_max)/float(runs) for j in range(steps)]
        rewards += [avg_reward]
        optAct += [percOptAct]

    return rewards,optAct
