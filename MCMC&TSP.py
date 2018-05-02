import math
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx

def fx(x):
    """distribution function"""
    y = 0.6 * (stats.beta.pdf(x,1,8)) + 0.4 * (stats.beta.pdf(x,9,1))
    return y

def MCMC_smp(p,q):
    """Sampling using MCMC method"""
    if (0 <= p <= 1):
        samples = []
        samples_acpt = []
        x = p
        samples.append(x)
        t = []
        for i in range (0,1000):
            y = stats.norm.rvs(loc = x,scale = q)
            samples.append(y)
            a = fx(y)
            b = fx(x)
            alpha = np.minimum(1,(a / b))
            acc = random.random();
            if acc <= alpha:
                samples_acpt.append(y)
                x = y
            else:
                samples_acpt.append(x)
        for i in range (0,len(samples_acpt)):
            t.append(i)
        plt.plot(t,samples_acpt)
        plt.title('Sample path')
        plt.xlabel('time')
        plt.ylabel('value')
        plt.show()
        plt.hist(samples_acpt,histtype = 'bar',edgecolor = 'black')
        plt.title('Samples generated')
        plt.xlabel('value')
        plt.ylabel('frequency(in number)')
        plt.show()
    else:
        print("The value input is not available.")

def fun1():
    """plot pdf of given distribution"""
    A = np.linspace(0,1.1,1000)
    X,Y = A, 0.6*(stats.beta.pdf(A,1,8))+ 0.4 * (stats.beta.pdf(A,9,1))
    plt.plot(X,Y)
    plt.title('The pdf of given distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def gx(x1,x2):
    """2D Scwefel function"""
    y = 418.9829 * 2 - ((x1 * np.sin(np.sqrt(np.abs(x1)))) + (x2 * np.sin(np.sqrt(np.abs(x2)))))
    return y


def MCMC_opt(q1,q2,t,mod):
    """Using simulated annealing algorithm to find global minimum"""
    fig = plt.figure()
    x1 = np.linspace(-500,500,10000)
    x2 = np.linspace(-500,500,10000)
    X1,X2 = np.meshgrid(x1,x2)
    Y = 418.9829 * 2 - ((X1 * np.sin(np.sqrt(np.abs(X1)))) + (X2 * np.sin(np.sqrt(np.abs(X2)))))
    ax = fig.gca(projection = '3d')
    ax.plot_surface(X1,X2,Y)
    plt.title('3D version of Scwefel function')
    plt.show()
    
    plt.contour(X1,X2,Y)
    plt.title('The contour plot of 2D Scwefel function')
    plt.xlabel('x1')
    plt.ylabel('x2')
    #plt.show()
    
    x1 = []
    x2 = []
    mini_value = []
    finn = []
    X1_route = []
    X2_route = []
    for k in range(0,100):
        if ((-500 <= q1 <= 500)and(-500 <= q2 <= 500)):
            x_1 = q1
            x_2 = q2
            x1.append(x_1)
            x2.append(x_2)
            T0 = 5000
            T = T0
            for i in range (0,t):
                if mod == 1:
                    T = T0 / (1 + 2000 * (np.log(1 + i)))         #log
                elif mod == 2:
                    T = T0 * np.power(0.99,i)                       #exponential
                elif mod == 3:
                    T = T0 / (1 + 0.05 * (np.square(i)))                  #polynomial
                else:
                    print("Invalid mode, exit.")
                    break

                z_1 = stats.norm.rvs(loc = x_1,scale = 166)
                z_2 = stats.norm.rvs(loc = x_2,scale = 166)
                if ((-500 <= z_1 <= 500)and(-500 <= z_2 <= 500)):
                    alpha = np.minimum(1,np.exp(-(gx(z_1,z_2) - gx(x_1,x_2)) / T))
                    acc = random.random();
                    if acc <= alpha:
                        x1.append(z_1)
                        x2.append(z_2)
                        x_1 = z_1
                        x_2 = z_2
                    else:
                        x1.append(x_1)
                        x2.append(x_2)
            X1_route.append(x1)
            X2_route.append(x2)
            finn.append(gx(x1[len(x1)-1],x2[len(x2)-1]))
            x1 = []
            x2 = []
    plt.plot(X1_route[finn.index(min(finn))],X2_route[finn.index(min(finn))],marker = 'o',mec = 'r',mfc = 'r')
    plt.show()
    plt.hist(finn,histtype = 'bar',edgecolor = 'black')
    plt.xlabel('The minimum value found')
    plt.ylabel('The frequency(in number)')
    if mod == 1:
        plt.title('Results come from log cooling schedule')
    elif mod == 2:
        plt.title('Results come from exponential cooling schedule')
    elif mod == 3:
        plt.title('Results come from polynomial cooling schedule')
    plt.show()

def dst_fun_old(x1,y1,x2,y2):
    """Destination function"""
    dist = np.square((x2 - x1),2) + np.square((y2 - y1),2)
    return dist

def dst_fun(City_x,City_y):
    """Destination distance function"""
    sum = 0
    for i in range(0,len(City_x) - 1):
        sum = sum + np.sqrt(np.square(City_x[i+1] - City_x[i]) + np.square(City_y[i+1] - City_y[i]))
    return sum

def TSP(t,m):
    """Traveling Salesman Problem using Simulated Annealing Problem"""            
    Cx = []
    Cy = []
    for i in range(0,m):
        Cx.append(random.randint(0,1000))
        Cy.append(random.randint(0,1000))
    T = t
    T0 = T
    i = 1
    d = []
    time = []
    for k in range(0,40000):
        T = T0 / (1 + 0.0000001 * (np.square(k))) 
        #T = T0 / (1 + 10 * (np.log(1 + k)))   log
        #T = T0 * np.power(0.99,k)             exponential
        d_old = dst_fun(Cx,Cy)
        a = b = 1
        while (a == b):
            a = random.randint(0,m - 1)
            b = random.randint(0,m - 1)
        Cx[a],Cx[b] = Cx[b],Cx[a]
        Cy[a],Cy[b] = Cy[b],Cy[a]
        d_new = dst_fun(Cx,Cy)
        alpha = np.minimum(1,np.exp(-((d_new - d_old) / T)))
        acc = random.random()
        if acc <= alpha:
            d.append(d_new)
        else:
            d.append(d_old)
            Cx[a],Cx[b] = Cx[b],Cx[a]
            Cy[a],Cy[b] = Cy[b],Cy[a]
            #print(d)
        #T = T * 0.99
    for k in range(0,len(d)):
        time.append(k)
    plt.plot(time,d)
    plt.title('The convegence path')
    plt.xlabel('time')
    plt.ylabel('total path length')
    plt.show()
    plt.plot(Cx,Cy,marker = 'o',mec = 'r',mfc = 'r')
    plt.title('Finally chosen route')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.show()
    print(d[len(d) - 1])

fun1()
p = input("Please input the initial value:")
q = input("Please input variance of proposal function:")
MCMC_smp(p,q)

m = input("Please input the initial temperature:")
n = input("Please input the number of cities:")
TSP(int(m),int(n))

r = input("Please input the iteration time:")
mod = input("Please input the model of cooling schedule(1 for log;2 for exp;3 for poly):")
MCMC_opt(0,0,int(r),int(mod))

        
    

