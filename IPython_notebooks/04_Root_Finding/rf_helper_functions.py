import numpy as np
import matplotlib.pyplot as plt
import sympy as syp

def correct_R30(Z,a0,r):
    a = 2.*((Z/(3*a0))**(3./2.))
    b =(1 - ((2*Z*r)/(3*a0))+((2*(Z*r)**2)/(27*(a0**2))))
    c = np.exp(-(Z*r)/(3*a0))
    sol = a*b*c
    return sol

def check_R30(func,Z,a,r):
    tol = 1.e-10
    yours = func(Z,a,r)
    correct =correct_R30(Z,a,r)
    diff = np.abs(yours-correct)
    if diff <=tol:
        print("CORRECT! Great job, you can continue to the next exercise")
    else:
        print("""ERROR: Your function does not produce the right result.
        Hints for troubleshooting:
        --> Are you reading in the variables in the correct order?
        --> Are there parentheses missing?""")
    return

def plotf_func(xminimum, xmaximum, f):
    x = np.linspace(xminimum, xmaximum, 101)
    y = f(x)
    yrange = np.max(y) - np.min(y)
    yminimum = np.min(y) - 0.1 * yrange
    ymaximum = np.max(y) + 0.1 * yrange
    fig = plt.figure(facecolor='white', figsize=(5, 5))
    ax = fig.add_subplot(111)
    plt.xlim(xminimum, xmaximum)
    plt.ylim(yminimum, ymaximum)
    plt.grid(True)
    plt.plot([xminimum, xmaximum], [0, 0],
             color='black', linestyle='-', linewidth=0.5)
    plt.plot([0, 0], [yminimum, ymaximum],
             color='black', linestyle='-', linewidth=0.5)
    plt.plot(x, y, color='black', linewidth=2.0)
    plt.xlabel(u'$x$')
    plt.ylabel(r'f($x$)')
    fig.set_tight_layout(False)
    plt.show()

    
def analytical_first_derivative(function, real_x):
    try:
        x = syp.Symbol('x')
        d_function = syp.diff(function(x), x)
        d_functional_function =  syp.lambdify(x, d_function, 'numpy')
        return d_functional_function(real_x)
    except:
        return 'No analytical gradient exsists'