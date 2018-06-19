import numpy as np
import sympy as syp
import matplotlib.pyplot as plt
import matplotlib
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
    x_data = np.linspace(xminimum, xmaximum, 101)
    try:
        y = f(x_data)
    except:
        x=syp.Symbol('x')
        f_temp=syp.lambdify(x,f(x),"numpy")
        y=f_temp(x_data)

    yrange = np.max(y) - np.min(y)
    yminimum = np.min(y) - 0.001*yrange
    ymaximum = np.max(y) + 0.001*yrange

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim(xminimum, xmaximum)
    plt.ylim(yminimum, ymaximum)

    plt.plot(x_data, y, color='black')
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
    x_data = np.linspace(xminimum, xmaximum, 101)
    try:
        y = f(x_data)
    except:
        x=syp.Symbol('x')
        f_temp=syp.lambdify(x,f(x),"numpy")
        y=f_temp(x_data)

    yrange = np.max(y) - np.min(y)
    yminimum = np.min(y) - 0.1*yrange
    ymaximum = np.max(y) + 0.1*yrange

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.xlim(xminimum, xmaximum)
    plt.ylim(yminimum, ymaximum)

    plt.grid(True)

    plt.plot(x_data, y, color='black')
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
    x_data = np.linspace(xminimum, xmaximum, 101)
    try:
        y = f(x_data)
    except:
        x=syp.Symbol('x')
        f_temp=syp.lambdify(x,f(x),"numpy")
        y=f_temp(x_data)

    yrange = np.max(y) - np.min(y)
    yminimum = np.min(y) - 0.1*yrange
    ymaximum = np.max(y) + 0.1*yrange

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.xlim(xminimum, xmaximum)
    plt.ylim(yminimum, ymaximum)

    plt.plot(x_data, y, color='black')
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
    x_data = np.linspace(xminimum, xmaximum, 101)
    try:
        y = f(x_data)
    except:
        x=syp.Symbol('x')
        f_temp=syp.lambdify(x,f(x),"numpy")
        y=f_temp(x_data)

    yrange = np.max(y) - np.min(y)
    yminimum = np.min(y) - 0.1*yrange
    ymaximum = np.max(y) + 0.1*yrange

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.xlim(xminimum, xmaximum)
    plt.ylim(yminimum, ymaximum)

    plt.grid(True)

    plt.plot(x_data, y, color='black')
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
    x_data = np.linspace(xminimum, xmaximum, 101)
    randomSamples=((xmaximum-xminimum)*np.random.uniform(size=n))+xminimum # returns a collection of random samples of size n within [a,b]

    try:
        y = f(x_data)
        valuesOfFunctions=f(randomSamples) # evaluate the function at each of the random samples.
    except:
        x=syp.Symbol('x')
        f_temp=syp.lambdify(x,f(x),"numpy")
        y=f_temp(x_data)
        valuesOfFunctions=f_temp(randomSamples) # evaluate the function at each of the random samples.

    yrange = np.max(y) - np.min(y)
    yminimum = np.min(y) - 0.1*yrange
    ymaximum = np.max(y) + 0.1*yrange

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.xlim(xminimum, xmaximum)
    plt.ylim(yminimum, ymaximum)


    plt.plot(x_data, y, color='black', linewidth=2.0)
    plt.plot([xminimum, xmaximum], [0, 0],color='black', linestyle='-', linewidth=0.5)
    plt.plot([0, 0], [yminimum, ymaximum] ,color='black', linestyle='-', linewidth=0.5)



    cmap = matplotlib.cm.get_cmap('jet')
    for j in range(len(randomSamples)):
        rand_color = cmap(np.random.random())
        plt.plot(randomSamples[j],  valuesOfFunctions[j],marker='.',color=rand_color,markersize=15)
    plt.xlabel(u'$x$')
    plt.ylabel(r'f($x$)')

    plt.show()

def analytical_integral(function, xminimum, xmaximum):
    x = syp.Symbol('x')
    # syp.integrate(function(x), x)
    return syp.integrate(function(x), (x, xminimum, xmaximum))
    print('Analytical integral exists')


def sympy2numpy_function_converter(f,variable):
    variable=syp.Symbol('variable')
    f_temp = syp.lambdify(variable,f(variable),"numpy")
    return f_temp

# plot_midpoint_func(0,2*np.pi,f,8)
