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
        optimalActions += [1+a]
    return reward, optimalActions, Q

def runSimpleBandit(R,steps,runs,eps,init=0,alpha=0,randomWalkSigma = 0):
    rewards = []
    optAct  = []
    avg_value = []
    idx_R_max = [1 + a for a in np.where(R == np.max(R))[0]]
    for e in range(len(eps)):
        avg_reward = [0 for i in range(steps)]
        percOptAct = [0 for i in range(steps)]
        value   = [0 for r in R]
        for i in range(runs):
            reward, optimalActions, Q = simpleBandit(R,steps,eps[e],init,alpha,randomWalkSigma)
            avg_reward = [avg_reward[j] + reward[j]/float(runs) for j in range(steps)]
            opt_actCount = [float(a in idx_R_max) for a in optimalActions]
            percOptAct = [percOptAct[j] + opt_actCount[j]/float(runs) for j in range(steps)]
            value = [value[j] + Q[j]/float(runs) for j in range(len(Q))]
        rewards += [avg_reward]
        optAct += [percOptAct]
        avg_value += [value]
    return rewards,optAct,avg_value


def unBiased_simpleBandit(R,steps,eps,init = 0,alpha=0,randomWalkSigma = 0):
    k = len(R)
    Q = []
    N = []
    for i in range(k):
        Q += [init]
        N += [0]
    idx_R_max = np.where(R == np.max(R))[0]
    reward = []
    optimalActions = []
    o = [0]
    beta = []
    for i in range(steps):
        o += [o[i] + alpha*(1-o[i])]
        beta += [alpha/o[i+1]]
        prob = np.random.uniform(0,1,1)[0]
        if prob >= eps:
            idx_Q_max = np.where(Q == np.max(Q))[0]
            a = int(np.random.uniform(idx_Q_max[0],idx_Q_max[-1]+1,1)[0])
        else:
            a = int(np.random.uniform(0,k,1)[0])
        Ri = Gaussian(R[a],1.0)[0]
        reward += [Ri]
        N[a] += 1
        Q[a] += beta[i]*(Ri - Q[a])
        if randomWalkSigma != 0:
            Q = [Q[s] + Gaussian(0,randomWalkSigma)[0] for s in range(k)]
        optimalActions += [1+a]
    return reward, optimalActions, Q, o, beta
