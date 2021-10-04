import numpy as np
import matplotlib.pyplot as plt

####Problem 1
#Define function for the ODE, as well as counter to count the 
#number of function evaluations
def f(x, y):
    f.counter += 1
    return y/(1+x**2)
f.counter = 0

#Define true function to plot with our estimation and look at the residuals
#to see which method is better. 
#When solving the given ODE we get dy/y = dx/1+x^2. This is solvable and gives the 
#solution y(x) = y_0 * e^(arctan(x) - arctan(x_0)), with x_0 = -20 and y_0 = y(-20) = 1
def true(x, y0, x0):
    sol =  y0*np.exp(np.arctan(x)-np.arctan(x0))
    return sol

#Using rk4 code defined in class
#updating y already in rk4_step
def rk4_step(fun,x,y,h):
    k1=fun(x,y)*h
    k2=h*fun(x+h/2,y+k1/2)
    k3=h*fun(x+h/2,y+k2/2)
    k4=h*fun(x+h,y+k3)
    dy=(k1+2*k2+2*k3+k4)/6
    return y+dy

#Defining global variables
#We set the number of steps to 199, so that we get the same number of function calls for both methods. 
#I found that when using 200 steps for rk4_step I got 796 function calls and when I used rk4_stepd with
#200/3 = 67 steps I got 792 function calls. To get rid of this difference I reduced the number of steps done
# by rk4_step by 1, so that I would get 796-4=792 function calls.
y0 = 1
nsteps = 199
#Defining x range
x = np.linspace(-20, 20, nsteps)
#Getting the true solution to the ODE
true_y = true(x, y0, x[0])

#Initializing my solution to the ODE
rk4_y = np.zeros(len(x))
rk4_y[0] = y0
#Getting all the y values using rk4_step
for i in range(len(x)-1):
    h = x[i+1] - x[i]
    new_y = rk4_step(f, x[i], rk4_y[i], h)
    rk4_y[i+1] = new_y

#Printing the total and per step number of function calls
print('Number of function calls for classic rk4 is:', f.counter)
print('Number of function evaluations per step is:', f.counter / (len(x)-1))

#Plotting my solution using rk4_step with the residuals
plt.plot(x, true_y, 'r')
plt.plot(x, rk4_y, 'b.')
plt.plot()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Classic RK4 ODE solver')
plt.show()

plt.plot(x, true_y - rk4_y, 'b.')
plt.plot(x, true_y*0, 'k')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.title('Residuals for Classic RK4 ODE solver')
plt.show()

#Resetting the function counter for the second method
f.counter = 0

#Defining the function for the RK4 half-step size method.
#We call the rk4 3 times, once with a step size of h and twice
# with a step size of h/2. We then compare the values of both approaches
# and define the difference as Delta. We use this delta to cancel out
#the leading-order error term in RK4. 
def rk4_stepd(fun, x, y, h):
    y1 = rk4_step(fun, x, y, h)
    y2_bis = rk4_step(fun, x, y, h/2)
    y2 = rk4_step(fun, x, y2_bis, h/2)
    Delta = y2 - y1
    return y2 + Delta/15

#As described above, the half-step size method calls the rk4 method 3 times. 
#So, in order to get the same number of function calls for both methods, we have to decrease
#the number of x values taken by 3 for the half-step size method.
#So we redefine nsteps to be 200/3 or 67. 
nsteps = round(200/3)
x = np.linspace(-20, 20, nsteps)

#Redefining the true solution of the ODE with the new x values
true_y = true(x, y0, x[0])

#Initializing my solution to the ODE. 
rk4d_y = np.zeros(len(x))
rk4d_y[0] = y0
#Getting all the y values using rk4_stepd
for i in range(len(x)-1):
    h = x[i+1] - x[i]
    new_yd = rk4_stepd(f, x[i], rk4d_y[i], h)
    rk4d_y[i+1] = new_yd

#Printing the total and per step number of function calls
print('Number of function calls for half-step size rk4 is:', f.counter)
print('Number of function evaluations per step is:', f.counter / (len(x)-1))

#Plotting my solution using rk4_stepd with the residuals
plt.plot(x, true_y, 'r')
plt.plot(x, rk4d_y, 'g.')
plt.plot()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Half-Step Size RK4 ODE solver')
plt.show()

plt.plot(x, true_y - rk4d_y, 'g.')
plt.plot(x, true_y*0, 'k')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.title('Residuals for Half-Step Size RK4 ODE solver')
plt.show()

####Problem 2




