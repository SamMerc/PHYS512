import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

###PROBLEM 1

def problem1(x, func, fivefunc, firstfunc):
    ##Input the x at which we measure delta, the function f(x) and it's 5th derivative. We also add
    ## the first derivative which we'll use to compare our method to the true derivative of f(x). 
    ##This will help us determine if dx is good enough
    eps = 2**-52
    fx = func(x)
    fx5 = fivefunc(x)
    #Defining dx according to result of 1.b
    dx = ((eps * fx)/fx5)**(1/5)
    #Calculating four points for numerical derivative
    fx1 = func(x + dx)
    fx2 = func(x - dx)
    fx3 = func(x + 2*dx)
    fx4 = func(x - 2*dx)
    deriv = (8*fx1 -8*fx2 - fx3 + fx4)/(12*dx)
    #Taking the absolute error between our estimated derivative and the true derivative
    error = abs(deriv-firstfunc(x))
    return error

def fun(x): 
    return np.exp(0.01*x)

def fun1(x):
    return 0.01*np.exp(0.01*x)

def fun5(x):
    return np.exp(0.01*x)*(0.01)**5

sol1=problem1(3, np.exp, np.exp, np.exp)
sol2=problem1(3, fun, fun5, fun1)
print('Error between estimated four point derivative and true derivative of exp(x) is:', sol1)
print('Error between estimated four point derivative and true derivative of exp(0.01*x) is:', sol2)

###PROBLEM 2

def ndiff(fun, x, full):
    #Define machine epsilon.
    eps = 2**-52
    #Define value for dx to calculate two points.
    dx = eps**(1/3)
    #recalculate dx so that it represents the smallest amount between x and the next representable number 
    #in double-point precision.
    x1 = x+dx 
    dx=x1-x
    #Calculate two points used in numerical derivative.
    fx1 = fun(x + dx)
    fx2 = fun(x - dx)
    deriv = (fx1 - fx2)/(2*dx)
    #We use the roundoff error as it gives an upper bound. We know that an estimate for 
    #the roundoff error is epsilon*f(x)/delta, as shown in class.
    error = (eps*fun(x))/dx
    if not full:
        return deriv, dx, error
    else : 
        return deriv
    
###PROBLEM 3

dat = np.loadtxt('lakeshore.txt')

def Lakeshore(V, data):
    #Extract temperature and voltage values from .txt file.
    Voltage=data[:, 1]
    Temperature=data[:, 0]
    #We order the data so that we can use the spline interpolation.
    #We zip the data first so that the the temperature and voltage 
    #values still match after sorting. 
    Volt, Temp = (list(t) for t in zip(*sorted(zip(Voltage, Temperature))))
    #Perform an Initial Spline interpolation
    spln=interpolate.splrep(Volt,Temp)
    T=interpolate.splev(V, spln).tolist()
    
    ##Error calculation using bootstrap method.
    
    #Get list of all indexes.
    index_range=[i for i in range(len(Volt))]
    #Define sample size and number of samples.
    sample_size=100
    sample_number=20
    #Initialize matrix for later.
    matrix=[]
    for i in range(sample_number):
        #Get list of indices for each sample.
        sample = np.random.choice(index_range, size=sample_size, replace=False)
        #For each sample get the voltage and temperature values associated
        #with each index.
        newV=[]
        newT=[]
        for ind in sample: 
            newV.append(Volt[ind])
            newT.append(Temp[ind])
        #Since we are going to do a spline interpolation on the points selected above, 
        #we need to sort the lists again so that spline works.
        newV, newT = (list(t) for t in zip(*sorted(zip(newV, newT))))
        #Re-interpolate for each sample
        spln_new=interpolate.splrep(newV, newT)
        BooT=interpolate.splev(V, spln_new).tolist()
        #We convert the new temperatures to lists in order to append them to the matrix.
        matrix.append(BooT)
    #After getting the temperatures for each sample and adding them to a matrix, 
    #we want to verticalize the matrix. This will allow us to get the variance values
    #for each temperature value by using the matrix's columns
    matrix=np.vstack(matrix)
    #We check to see if the input is an array or not and calculate the standard deviation 
    #of each column to get the temperature error
    if isinstance(V, float):
        error=np.std(matrix)
    else : 
        error=[np.std(matrix[:, i]) for i in range(len(V))]
    return T, error

solution=Lakeshore(np.array([0.6, 0.8, 1.2, 1.4]), dat)
mean_err=np.mean(solution[1])
print("The average error on the temperature values is:", mean_err)


###PROBLEM 4
xx=np.linspace(-np.pi/2, np.pi/2, 20)
yy=np.cos(xx)
m1=len(yy)//2
n1=len(yy)-m1-1

xxx=np.linspace(-1, 1, 20)
yyy=[]
for i in xxx:
    yyy.append(1/(1+i**2))
m2=len(yyy)//2
n2=len(yyy)-m2-1

##Using the rateval and ratfit function defined in class for the ratioanl function interpolation.
def rateval(x,p,q):
    top=0
    for i,par in enumerate(p):
        top=top+par*x**i
    bot=1
    for i,par in enumerate(q):
        bot=bot+par*x**(i+1)
    return top/bot

def ratfit(y,x,n,m):
    npt=len(x)
    assert(len(y)==npt)
    assert(n>=0)
    assert(m>=0)
    assert(n+1+m==npt)

    top_mat=np.empty([npt,n+1])
    bot_mat=np.empty([npt,m])
    for i in range(n+1):
        top_mat[:,i]=x**i
    for i in range(m):
        bot_mat[:,i]=y*x**(i+1)
    mat=np.hstack([top_mat,-bot_mat])
    pars=np.linalg.inv(mat)@y
    p=pars[:n+1]
    q=pars[n+1:]
    return mat,p,q

def spline(x, y):
    ##Spline interpolation
    x, y = (list(t) for t in zip(*sorted(zip(x, y))))
    spln=interpolate.splrep(x, y)
    newy1=interpolate.splev(x, spln)
    accur1 = np.std(newy1-y)
    return newy1, accur1
def poly(x, y):
    ##Poly interpolation
    p=np.polyfit(x,y,5)
    newy2=np.polyval(p, x)
    accur2 = np.std(newy2-y)
    return newy2, accur2
def rational(x, y, m, n):
    ##Rational interpolation
    mat, p, q=ratfit(y, x, n, m)
    newy3 = rateval(x, p, q)
    accur3 = np.std(newy3-y)
    return newy3, accur3

y1, a1 = poly(xx, yy)
y2, a2 = spline(xx, yy)
y3, a3 = rational(xx, yy, m1, n1)
print("The accuracy of polynomial interpolation for cos is:", a1)
print("The accuracy of spline interpolation for cos is:", a2)
print("The accuracy of rational function interpolation for cos is:", a3)

y4, a4 = poly(xxx, yyy)
y5, a5 = spline(xxx, yyy)
y6, a6 = rational(xxx, yyy, m2, n2)
print("The accuracy of polynomial interpolation for Lorentzian is:", a4)
print("The accuracy of spline interpolation for Lorentzian is:", a5)
print("The accuracy of rational function interpolation for Lorentzian is:", a6)

##We expect the Lorentz function to be well interpolated by rational functions between -1 and 1, since that is the range
##where to function is well-described by a Taylor series. However, we observe that rational interpolation performs the worst
##among the three methods used, when using np.linalg.inv. 
##Increasing the order gives a better accuracy, however it still remains orders of magnitude higher than the other methods. When
##we switch to np.linalg.pinv there is a noticeable improvement.In this case, rational interpolation outperforms polynomial
##interpolation but not spline. 
##Looking at p and q, we see that the eigenvalues are pretty small for both of them, even when increasing the order. This means ##that when we call np.linalg.inv the values become really large and cause the accuracy to drop. However when we implement 
##np.linalg.pinv the small eigenvalues are set to 0 when the matrix is inverted, allowing for a higher accuracy.