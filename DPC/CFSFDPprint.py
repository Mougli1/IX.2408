import numpy as np
from matplotlib import pyplot as plt


def CFSFDPprint(X, clt, rho, delta, seeds, Pretty=True):
    unique_labels = set(range(0, int(np.max(clt))))
    colors = [plt.cm.rainbow(each) for each in np.linspace(0, 1, len(unique_labels) + 1)]

    fig, ax = plt.subplots(figsize=(12, 12))
    if Pretty == True:
        for i in range(len(X)):
            k = int(clt[i])
            if k == 0 or k == -1:
                plt.scatter(rho[i], delta[i], color="black", alpha=0.3)
            else:
                plt.scatter(rho[i], delta[i], color=colors[k], alpha=1)
    else:
        ax.scatter(rho, delta, c=clt, cmap='rainbow')
        for i in range(len(seeds)):
            ax.scatter(rho[seeds[i]], delta[seeds[i]], s=70, color='k')
    plt.xlabel("Rho")
    plt.ylabel("delta")
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 12))
    if Pretty == True:
        for i in range(len(X)):
            k = int(clt[i])
            if k == 0 or k == -1:
                plt.scatter(X[i][0], X[i][1], color="black", alpha=0.3)
            else:
                plt.scatter(X[i][0], X[i][1], color=colors[k], alpha=1)
    else:
        ax.scatter(X[:, 0], X[:, 1])
    for i in range(len(seeds)):
        ax.scatter(X[seeds[i]][0], X[seeds[i]][1], s=70, color='k')
    plt.show()

    arg_gamma = np.argsort(rho * delta)
    fig, ax = plt.subplots(figsize=(12, 12))
    if Pretty == True:
        for i in range(len(arg_gamma)):
            nb = arg_gamma[len(arg_gamma) - i - 1]
            k = int(clt[nb])
            if k == 0 or k == -1:
                plt.scatter(i, rho[nb] * delta[nb], color="black", alpha=0.3)
            else:
                plt.scatter(i, rho[nb] * delta[nb], color=colors[k], alpha=1)
    plt.xlabel("N")
    plt.ylabel("Gamma=Rho*delta")
    plt.show()
