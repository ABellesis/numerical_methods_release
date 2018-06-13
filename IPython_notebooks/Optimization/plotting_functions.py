import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import numpy as np
from  scipy.optimize import line_search as linesearch
import scipy as sp


def plot_func(xminimum,xmaximum,yminimum,ymaximum,steps_x,steps_y,f):
    fig = plt.figure(figsize=(14,6.6))
    # fig = plt.figure()
    X=np.linspace(xminimum,xmaximum,100)
    Y=np.linspace(yminimum,ymaximum,100)
    X, Y = np.meshgrid(X, Y)
    points=np.array((X,Y))
    Z=f(points)
    Zrange= np.max(Z) - np.min(Z)
    zminimum = np.min(Z) - 0.1*Zrange
    zmaximum = np.max(Z) + 0.1*Zrange
    ax = plt.subplot(121, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,alpha=0.45, linewidth=0, antialiased=True, cmap = cm.jet)
    cset = ax.contour(X, Y, Z, zdir='Z', offset = -1, cmap=cm.jet)
    ax.set_xlabel("X", linespacing = 3.2)
    ax.set_zlabel("Z", linespacing=3.2)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.rcParams['ytick.major.pad']='8'
    ax.set_xlim(xminimum, xmaximum)
    ax.set_ylim(yminimum, ymaximum)
    ax.set_zlim(zminimum,zmaximum)

    ax2=plt.subplot(122)
    plt.contour(X,Y,Z, 15, colors ='k')
    plt.contourf(X,Y,Z, 15, cmap=cm.jet,vmax=abs(Z).max(), vmin=-abs(Z).max())
    plt.plot(steps_x,steps_y, marker = 'o',markerfacecolor='white',markeredgecolor='black',color='white')
    ax.scatter(steps_x,steps_y,f((steps_x,steps_y)),color='black',marker='o')
    ax2.set_xlim(steps_x[len(steps_x)-1]-0.5, steps_x[len(steps_x)-1]+0.5)
    ax2.set_ylim(steps_y[len(steps_y)-1]-0.5, steps_y[len(steps_y)-1]+0.5)
    fig.set_tight_layout(True)
    # plt.savefig("testnewton.png",bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("This file is contains additional plotting functions. It is not meant to be run independently")
