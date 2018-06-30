import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sympy as syp
import scipy as sp
from scipy import integrate
from matplotlib.widgets import Slider, Button
import scipy.constants as sc
syp.init_printing(use_unicode=False, wrap_line=False, no_global=True)
import warnings

def midpoint(a,b,f):  # midpont rule evaluation of the function, called in Cmidpoint below
    mp=(a+b)/2.0
    return (b-a)*f(mp)
def Trapezoid(f,a,b,n):
    h=(b-a)/float(n)  # determines the width of the trapezoid for which the evaluation of f(x) will be calculated
    estimate=0
    estimate+=f(a)/2.0   # the initial evaluation of f(x)
    for i in range(1,n):
        estimate+=f(a+i*h) # sums f(x) each iteration, resulting in an estimate of the integration
    estimate+=f(b)/2.0  # the final evaluation of f(x)
    return estimate

def plot_midpoint_func(xminimum,xmaximum,f,subsections):
    x = np.linspace(xminimum,xmaximum,101)
    y = f(x)

    yrange = np.max(y) - np.min(y)
    yminimum = np.min(y) - 0.001*yrange
    ymaximum = np.max(y) + 0.001*yrange

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    plt.xlim(xminimum, xmaximum)
    plt.ylim(yminimum, ymaximum)

    plt.plot(x, y, color='black')
    plt.plot([xminimum, xmaximum], [0, 0],color='black', linestyle='-')
    plt.plot([0, 0], [yminimum, ymaximum],color='black', linestyle='-')
    # red lines

    lines_x = np.linspace(xminimum,xmaximum,subsections+1)
    for i in range(len(lines_x)-1):
        midpoint = (lines_x[i+1] + lines_x[i]) / 2.0
        plt.plot([lines_x[i], lines_x[i]], [0, f(midpoint)],color='red', linestyle='-')
        plt.plot([lines_x[i+1], lines_x[i+1]], [0, f(midpoint)],color='red', linestyle='-')
        plt.plot([midpoint, midpoint], [0, f(midpoint)],color='red', linestyle='--')
        plt.plot([lines_x[i], lines_x[i+1]], [f(midpoint), f(midpoint)],color='red', linestyle='-')

    plt.xlabel(u'$x$')
    plt.ylabel(r'f($x$)')
    plt.show()

def plot_trapezoidal_func(xminimum,xmaximum,f,subsections):
    x = np.linspace(xminimum,xmaximum,100)
    y = f(x)

    yrange = np.max(y) - np.min(y)
    yminimum = np.min(y) - 0.1*yrange
    ymaximum = np.max(y) + 0.1*yrange

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    plt.xlim(xminimum, xmaximum)
    plt.ylim(yminimum, ymaximum)

    plt.grid(True)

    plt.plot(x, y, color='black')
    plt.plot([xminimum, xmaximum], [0, 0],color='black', linestyle='-')
    plt.plot([0, 0], [yminimum, ymaximum],color='black', linestyle='-')
    # red lines
    lines_x = np.linspace(xminimum,xmaximum,int(subsections+1))
    for i in range(len(lines_x)-1):
        plt.plot([lines_x[i], lines_x[i]], [0, f(lines_x[i])],color='red', linestyle='-')
        plt.plot([lines_x[i+1], lines_x[i+1]], [0, f(lines_x[i+1])],color='red', linestyle='-')
        plt.plot([lines_x[i], lines_x[i+1]], [f(lines_x[i]), f(lines_x[i+1])],color='red', linestyle='-')


    plt.xlabel(u'$x$')
    plt.ylabel(r'f($x$)')

    plt.show()

def plot_simpsons_func(xminimum,xmaximum,f,parabola):
    x = np.linspace(xminimum,xmaximum,100)
    y = f(x)

    yrange = np.max(y) - np.min(y)
    yminimum = np.min(y) - 0.1*yrange
    ymaximum = np.max(y) + 0.1*yrange

    fig = plt.figure(facecolor='white', figsize=(7, 7))
    ax = fig.add_subplot(111)

    plt.xlim(xminimum, xmaximum)
    plt.ylim(yminimum, ymaximum)

    plt.plot(x, y, color='black')
    plt.plot([xminimum, xmaximum], [0, 0],color='black', linestyle='-')
    plt.plot([0, 0], [yminimum, ymaximum] ,color='black', linestyle='-')


    cmap = matplotlib.cm.get_cmap('jet')
    lines_x = np.linspace(xminimum,xmaximum,parabola+1)
    for i in range(len(lines_x)-1):
        rand_color = cmap(np.random.random())
        plt.plot([lines_x[i], lines_x[i]], [f(lines_x[i]), f(lines_x[i])],color='red', linestyle='-',marker ='o',markersize=12)
        plt.plot([lines_x[i+1],lines_x[i+1]], [f(lines_x[i+1]), f(lines_x[i+1])],color='red', linestyle='-',marker ='o',markersize=12)

        midpoint = (lines_x[i]+lines_x[i+1])/2.
        plt.plot([midpoint, midpoint], [f(midpoint), f(midpoint)],color='blue', linestyle='-',marker ='o',markersize=20)

        #plotting the parabola
        x_data=[lines_x[i],midpoint,lines_x[i+1]]
        y_data=[f(lines_x[i]),f(midpoint),f(lines_x[i+1])]
        z = np.polyfit(x_data,y_data,2)
        fn = np.poly1d(z)
        x_new=np.linspace(lines_x[i],lines_x[i+1],50)
        plt.plot(x_new, fn(x_new),color=rand_color, linestyle='--')

    plt.xlabel(u'$x$')
    plt.ylabel(r'f($x$)')
    plt.show()

def plot_gaussian_func(xminimum,xmaximum,f,n):
    x = np.linspace(xminimum,xmaximum,100)
    y = f(x)

    yrange = np.max(y) - np.min(y)
    yminimum = np.min(y) - 0.1*yrange
    ymaximum = np.max(y) + 0.1*yrange

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    plt.xlim(xminimum, xmaximum)
    plt.ylim(yminimum, ymaximum)

    plt.grid(True)

    plt.plot(x, y, color='black')
    plt.plot([xminimum, xmaximum], [0, 0],color='black', linestyle='-')
    plt.plot([0, 0], [yminimum, ymaximum] ,color='black', linestyle='-')

    x_i,w=np.polynomial.legendre.leggauss(n) # determination of optimal points and weights for evaluation based on legendre polynomials
    x_i=((xmaximum-xminimum)/2.0)*x_i+((xmaximum+xminimum)/(2.0)) # change of variables for [-1,1] to [a,b]
    for j in range(n):
        plt.plot(x_i[j], f(x_i[j]),marker='.', color='red',markersize=100*w[j])
    plt.xlabel(u'$x$')
    plt.ylabel(r'f($x$)')
    plt.show()

def plot_MonteCarlo_func(xminimum,xmaximum,f,n):
    x = np.linspace(xminimum,xmaximum,100)
    y = f(x)


    yrange = np.max(y) - np.min(y)
    yminimum = np.min(y) - 0.1*yrange
    ymaximum = np.max(y) + 0.1*yrange

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    plt.xlim(xminimum, xmaximum)
    plt.ylim(yminimum, ymaximum)


    plt.plot(x, y, color='black', linewidth=2.0)
    plt.plot([xminimum, xmaximum], [0, 0],color='black', linestyle='-', linewidth=0.5)
    plt.plot([0, 0], [yminimum, ymaximum] ,color='black', linestyle='-', linewidth=0.5)

    randomSamples=((xmaximum-xminimum)*np.random.uniform(size=n))+ xminimum # returns a collection of random samples of size n within [a,b]
    valuesOfFunctions=f(randomSamples) # evaluate the function at each of the random samples.



    cmap = matplotlib.cm.get_cmap('jet')
    for j in range(len(randomSamples)):
        rand_color = cmap(np.random.random())
        plt.plot(randomSamples[j],  valuesOfFunctions[j],marker='.',color=rand_color,markersize=15)
    plt.xlabel(u'$x$')
    plt.ylabel(r'f($x$)')

    plt.show()

##############################################################################
############################### Differentiation ##############################
##############################################################################

def plot_fdiff_func(xminimum, xmaximum, f, derivative_f, a, da):
    x_data = np.linspace(xminimum, xmaximum, 1001)
    try:
        x=syp.Symbol('x')
        f_temp=syp.lambdify(x,f(x),np)
        y=f_temp(x_data)
    except:
        y = f(x_data)
        print("y for the try statement  {}".format(y))
    #fd_step_size_result = fd_step_size_check(f,da,a)
    #print(fd_step_size_result['message'])
    yrange = np.max(y) - np.min(y)
    yminimum = np.min(y) - 0.1 * yrange
    ymaximum = np.max(y) + 0.1 * yrange
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    plt.xlim(xminimum, xmaximum)
    plt.ylim(yminimum, ymaximum)
    plt.grid(True)
    plt.plot(x_data, y, color='black', linewidth=2.0)
    plt.plot(x_data, tangent_line(a,f, derivative_f, x_data), color='blue', linestyle='-', linewidth=6.0,label = 'true')
    plt.plot([(a + da), (a + da)], [f((a + da)), f((a + da))],
             color='red', markersize=20, marker='.')
    plt.plot([a, a], [f(a), f(a)], color='blue', markersize=20, marker='.')

    plt.plot(x_data, secant_line(f, a, (a + da), x_data), color='red', linewidth=2.0,label ='approximation')
    plt.legend()
    plt.xlabel(u'$x$')
    plt.ylabel(r'f($x$)')

    fig.set_tight_layout(True)

    plt.show()

def slope(x1, x2, y1, y2):
    rise = y2 - y1
    run = x2 - x1
    slope = rise / run
    return slope

def intercept(x1, x2, y1, y2):
    m = slope(x1, x2, y1, y2)
    b = y2 - (m * x2)
    return m, b

def secant_line(f, x1, x2, x):
    m, b = intercept(x1, x2, f(x1), f(x2))
    secant_line = (m * x) + b
    return secant_line

def tangent_line(a,f, derivative_f,x):
    return f(a) + derivative_f(f,a)*(x - a)

def plot_cdiff_func(xminimum, xmaximum, f,derivative_f, a, da):
    x_data = np.linspace(xminimum, xmaximum, 101)
    try:
        y = f(x_data)
    except:
        x=syp.Symbol('x')
        f_temp=syp.lambdify(x,f(x),"numpy")
        y=f_temp(x_data)

    #cd_step_size_result = cd_step_size_check(f,da,a)
    #print(cd_step_size_result['message'])
    yrange = np.max(y) - np.min(y)
    yminimum = np.min(y) - 0.1 * yrange
    ymaximum = np.max(y) + 0.1 * yrange
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    plt.xlim(xminimum, xmaximum)
    plt.ylim(yminimum, ymaximum)
    plt.grid(True)
    plt.plot([a, a], [f(a), f(a)], color='blue', markersize=20, marker='.')
    plt.plot(x_data, tangent_line(1,f,derivative_f, x_data), color='blue', linestyle='-', linewidth=6.0,label = 'true')
    plt.plot(x_data, y, color='black', linewidth=2.0)
    plt.plot(x_data, secant_line(f, (a - da), (a + da), x_data),
             color='red', linewidth=2.0, label = 'approximate')
    plt.plot([a + da, a + da], [f(a + da), f(a + da)],
             color='red',markersize=20, marker='.')
    plt.plot([a - da, a - da], [f(a - da), f(a - da)],
             color='red', markersize=20, marker='.')
    lgd = plt.legend()
    plt.xlabel(u'$x$')
    plt.ylabel(r'f($x$)')

    fig.set_tight_layout(True)

    plt.show()

def analytical_first_derivative(function, real_x):
    try:
        x = syp.Symbol('x')
        d_function = syp.diff(function(x), x)
        d_functional_function =  syp.lambdify(x, d_function, 'numpy')
        return d_functional_function(real_x)
    except:
        return 'No analytical gradient exsists'

def analytical_second_derivative(function, real_x):
    try:
        x = syp.Symbol('x')
        second_d_function = syp.diff(function(x), x, 2)
        second_d_functional_function =  syp.lambdify(x, second_d_function, 'numpy')
        return second_d_functional_function(real_x)
    except:
        print('No analytical gradient exsists')
        return

def analytical_third_derivative(function, real_x):
    try:
        x = syp.Symbol('x')
        second_d_function = syp.diff(function(x), x, 3)
        second_d_functional_function =  syp.lambdify(x, second_d_function, 'numpy')
        return second_d_functional_function(real_x)
    except:
        print('No analytical gradient exsists')
        return
    
def fd_step_size_check(f,h,x):
    step_size_min = np.finfo(float).eps
    try:
        warnings.filterwarnings('error')
        step_size_max = 2*(np.sqrt(step_size_min))/analytical_third_derivative(f,x)
    except:
        warnings.filterwarnings('default')
        print('Warning: no third derivative, using default max step size of 1e-5 as comparison')
        step_size_max = 1e-5
    ss_result = False
    if h < step_size_max and h > step_size_min:
        ss_result = True
        ss_message = ""

    else:
        ss_result = False
        if h >= step_size_max:
            ss_message = """
            =========================================================
                                   WARNING:
                          The step size is too large
                       The answer below is not reliable
                        Truncation error will dominate
                         Please input a smaller value
            =========================================================
            """

        if h <= step_size_min:
            ss_message = """
            =========================================================
                                   WARNING:
                          The step size is too small
                       The answer below is not reliable
                         Round-off error will dominate
                         Please input a larger value
            =========================================================
            """
    return {'result':ss_result, 'message':ss_message}

def cd_step_size_check(f,h,x):
    step_size_min = np.finfo(float).eps
    try:
        warnings.filterwarnings('error')
        step_size_max =  ((3*step_size_min)/analytical_third_derivative(f,x))**1/3
    except:
        warnings.filterwarnings('default')
        print('Warning: no third derivative, using default max step size of 1e-5 as comparison')
        step_size_max = 1e-5
    ss_result = False
    if h < step_size_max and h > step_size_min:
        ss_result = True
        ss_message = ""

    else:
        ss_result = False
        if h >= step_size_max:
            ss_message = """
            =========================================================
                                   WARNING:
                          The step size is too large
                       The answer below is not reliable
                        Truncation error will dominate
                         Please input a smaller value
            =========================================================
            """

        if h <= step_size_min:
            ss_message = """
            =========================================================
                                   WARNING:
                          The step size is too small
                       The answer below is not reliable
                         Round-off error will dominate
                         Please input a larger value
            =========================================================
            """
    return {'result':ss_result, 'message':ss_message}


def regression_plot(x,y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(True)
    yrange = np.max(y) - np.min(y)
    yminimum = np.min(y) - 0.1 * yrange
    ymaximum = np.max(y) + 0.1 * yrange
    xrange = np.max(x) - np.min(x)
    xminimum = np.min(x) - 0.1 * xrange
    xmaximum = np.max(x) + 0.1 * xrange
    plt.ylim(yminimum,ymaximum)
    plt.xlim(xminimum,xmaximum)
    plt.xlabel('1/Temperature')

    ax.scatter(x,y)
    fit = np.polyfit(x,y,1)
    fit_fn = np.poly1d(fit)
    plot_range=np.linspace(min(x),max(x),100)
    plt.ylabel('$\lambda$')
    ax.plot(plot_range, fit_fn(plot_range),label="y = {:.2e}(1/T)+{:3.2f}".format(fit[0],fit[1]))
    ax.legend()


def question_one_check(ans=0):
        if ans == 3:
            print("Correct! round-off error is the result of the way numbers\n are stored in the computer! If the step size we use is too small, round-off errors \ncan be blown up in the division involved for differentiation.")
        elif ans ==0:
            print("Please input a valid response")
        else:
            print("Incorrect. Please try again.")

def question_two_check(ans=0):

        if ans == 4:
            print("Correct! By truncating the Taylor series, we are losing information of higher derivatives. This loss can cause inaccuracies in the approximations of derivatives. This notation represents where the Taylor series is truncated.")
        elif ans ==0:
            print("Please input a valid response")
        else:
            print("Incorrect. Please try again.")
