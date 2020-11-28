import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal

def f(t,Y,m1):
    G=6.674-11#Constante de gravitación universal
    xprime = Y[2]
    yprime = Y[3]
    vxprime = Y[0]*G*m1/np.power(Y[0]**2+Y[1]**2,3/2)
    vyprime = Y[1]*G*m1/np.power(Y[0]**2+Y[1]**2,3/2)
    return np.array([xprime,yprime,vxprime,vyprime])

def rungeKuttaTwo(tiempoTotal,n,m1,t,Y,f):
    h=tiempoTotal/n #tamaÃ±o de paso
    solTiempo=np.zeros(n)#declarar un vector de tiempo compuesto por ceros 
    solY=np.zeros((n,4)) #declarar un vector soluciÃ³n Y compuesto por n vectores de 4 elementos 
    
    for i in range(n):
        solTiempo[i]=t
        solY[i]=Y
        k0=f(t,Y,m1)
        k1=f(t+h/2,Y+h*k0/2,m1)
        k2=f(t+h/2,Y+h*k1/2,m1)
        k3=f(t+h,Y+h*k2,m1)
        DY=h/6*(k0+2*k1+2*k2+k3)
    
        t=t+h #incremento en t
        Y=Y+DY #incremento en el vector Y
        #print(Y,DY)
    return solY[:,0],solY[:,1],solY[:,2],solY[:,3],solTiempo

def CometPopulation(x,y):
    xmin=min(x)-1.5
    xmax=max(x)+1.5
    ymin=min(y)-1.5
    ymax=max(y)+1.5
    A=np.zeros((1000,2))
    for i in range(0,1000,2):
        u=np.random.uniform()
        if(u<0.5):
            y=np.random.rand()*(ymax-ymin)+ymin
            A[i]=xmin,y
            A[i+1]=xmax,y
        else:
            x=np.random.rand()*(xmax-xmin)+xmin
            A[i]=x,ymin
            A[i+1]=x,ymax
    return A

def prob1(x,m,y,r):
    return np.exp(-8.8*m*y*r**(-1)/x)

def velocity(vx,vy,sigma=0.62):
    return -(vx**2+vy**2)/(2*sigma**2)


def metropolisHastingUniform(alpha1,n,f):
    x1 = np.zeros((n,2))
    x1[0] = [0.01,0.1]
    acept = 0
    for i in range(0, n-1):
        y = x1[i]+[np.random.uniform(-alpha1, alpha1),np.random.uniform(-alpha1, alpha1)]
        if np.random.rand() < min(1, f(y[0],y[1])/f(x1[i][0],x1[i][1])):
            x1[i+1] = y
            acept += 1
        else:
            x1[i+1] = x1[i]
    return x1,acept/n*100

def EDKepler(t,Y,masa):
    G=6.674-11#Constante de gravitación universal
    solucion=np.zeros(len(Y))#Se inicializa el vector solución 
    for i in range(0,int(len(Y)/2)):
        solucion[i]=Y[i+int(len(Y)/2)]#Se resuelve para la posicion
        
    for i in range(int(len(Y)/2),len(Y),2):
        sum1=0
        sum2=0
        pos=i-int(len(Y)/2)
        for j in range(0,len(masa)):
            if(int((i-int(len(Y)/2))/2)!=j):
                sum1=sum1+masa[j]*G*(Y[pos]-Y[2*j])/np.power((Y[pos]-Y[2*j])**2+(Y[pos+1]-Y[2*j+1])**2,3/2) #Se realiza la suma de los terminos expresados en la ecuacion diferencial
                sum2=sum2+masa[j]*G*(Y[pos+1]-Y[2*j+1])/np.power((Y[pos]-Y[2*j])**2+(Y[pos+1]-Y[2*j+1])**2,3/2) 
        solucion[i]=sum1 #Se resuelve para la velocidad
        solucion[i+1]=sum2 #Se resuelve para la velocidad
    return solucion #Retorna el vector solucion
def solver(tiempoTotal,n,t,Y,masa,f):
    h=tiempoTotal/n #tamaño de paso
    solTiempo=np.zeros(n)#declarar un vector de tiempo compuesto por ceros 
    solY=np.zeros((n,np.size(Y))) #declarar un vector solución Y compuesto por n vectores de 4 elementos 
    
    for i in range(n):
        solTiempo[i]=t
        solY[i]=Y
        k0=f(t,Y,masa)
        k1=f(t+h/2,Y+h*k0/2,masa)
        k2=f(t+h/2,Y+h*k1/2,masa)
        k3=f(t+h,Y+h*k2,masa)
        DY=h/6*(k0+2*k1+2*k2+k3)
    
        t=t+h #incremento en t
        Y=Y+DY #incremento en el vector Y
        #print(Y,DY)
    return solY,solTiempo

def cometAppears(ProbabilidadEscape):
    c=0
    k=2
    while(k!=1 and c!=500):
        k=np.random.choice(2, 1, p=[1-ProbabilidadEscape[c],ProbabilidadEscape[c]])
        c=c+1
    return c
