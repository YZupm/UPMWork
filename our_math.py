import numpy as np

# 1. Calibracion del sensor principal de empuje (Voltaje [V] vs Fuerza [kN])
voltaje_sensor = np.array([0.0, 1.25, 2.45, 3.80, 5.10])
fuerza_real = np.array([0.0, 15.2, 31.0, 48.5, 65.3])
# 2. Reacciones en la plataforma de lanzamiento (Coordenadas apoyos y pesos)
# Matriz de geometria estatica (distancias en m) para el balance de masas
matriz_estatica = np.array([
[1.0, 1.0, 1.0],
# Suma de fuerzas Z
[0.0, 1.5,-1.5],
# Momentos en X
[-2.0, 1.0, 1.0]
# Momentos en Y
])
vector_cargas = np.array([25000.0, 1500.0,-800.0]) # [N, Nm, Nm]
# 3. Datos de Atmosfera Estandar (Altitud [m], Temperatura [K], Presion [Pa])
h_atm = np.array([0, 2000, 4000, 6000, 8000, 10000, 12000, 14000])
T_atm = np.array([288.15, 275.15, 262.15, 249.15, 236.15, 223.15, 216.65,
216.65])
P_atm = np.array([101325, 79501, 61660, 47217, 35651, 26499, 19399, 14170])
# 4. Telemetria de vuelo: Tiempo [s] y Velocidad [m/s]
t_vuelo = np.linspace(0, 60, 1200) # Muestreo a 20 Hz
# (Perfil simulado de velocidad de un cohete)
v_vuelo = 800 * np.sin(np.pi * t_vuelo / 60) * np.exp(-0.02 * t_vuelo)

from math import prod
from numpy import dot, zeros, arange, newaxis, matmul, transpose, allclose, shape, ndim
from numpy.linalg import cond, norm


#BALANCE DE MASAS#
from numpy import zeros
def my_Gauss(A, b):
    N = len(b)
    x = zeros(N)

    for i in range(N):
      factores = A[i+1:, i] / A[i, i]   #está vectorizado al aplicar el factor y la resta sobre toda la fila cuantos menos fors mejor
      A[i+1:, :] = A[i+1:, :] - factores[:, None] * A[i, :] #el none sirve para generar la dimension de la columna pasando de (3, ) a (3,1) es lo mismo que el newaxis
      b[i+1:] = b[i+1:] - factores * b[i]     

    for i in range(N-1, -1, -1):
        x[i]=(b[i]-dot(A[i, i+1:], x[i+1:]))/A[i,i]       
    return x

def my_LU(A,b):
   N = len(b)
   x = zeros(N)
   y = zeros(N)
   if cond(A)<10**12 and len(A)==len(b):  
      L, U = zeros((N,N)), zeros((N,N))
      L[arange(N), arange(N)] = 1
      U[arange(N), arange(N)] = A [arange(N), arange(N)]
      for k in range(N):
          U[k, k:] = A[k, k:] - dot(L[k, :k], U[:k, k:])
          if U[k, k] == 0:
               raise ValueError("Cero; Necesitamos pivote")
          L[k + 1:, k] = (A[k + 1:, k] - dot(L[k + 1:, :k], U[:k, k]))/ U[k, k]
   else: raise ValueError("Validez comprometida o no cuadran las dimensiones")

   if not allclose(matmul(L,U),A):
        raise ValueError("Error en el proceso")
   else: 
      for i in range (N):
         y[i] = b[i]-dot(L[i,:i] , y[:i])
      for i in range(N - 1, -1, -1):
         x[i] = (y[i] - dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
   return x

def my_Jacobi (A,b):
    max_iter = 1000
    tol=10**(-16)
    N=len(b)
    x2,x1,R = zeros(N),zeros(N),zeros(N)
    LU = zeros((N,N))
    LU[:,:]=A[:,:]
    for i in range (N):
        LU [i,i]=0
    diagonal = A[arange(N), arange(N)]

    if np.any(diagonal == 0):
        raise ValueError("La diagonal contiene ceros")
    
    else:
        for _ in range(max_iter):
            x2 = (b - matmul(LU, x1)) / diagonal
            if norm(x2 - x1) < tol:
                return x2
            x1 = x2.copy()
        return x1

def my_Gauss_Seidel (A,b):
    max_iter = 1000
    tol=10**(-16)
    N=len(b)
    x2,x1,R = zeros(N),zeros(N),zeros(N)
    diagonal = A[arange(N), arange(N)]
    
    if np.any(diagonal == 0):
            raise ValueError("La diagonal contiene ceros")
    for _ in range(max_iter):
        for i in range(N):
                R[i]=b[i]-dot(A[i,i+1:], x1[i+1:])
        for i in range (N):
                x2[i] = ((R[i]-dot(A[i,:i], x2[:i]))/diagonal[i])
        if norm(x2 - x1) < tol:
            return x2
        x1 = x2.copy()

    return x1

#CALIBRACIÓN DE MOTOR#

def my_Vandermonde(n,m,array):
    N=max(n,m)
    x = array
    A = x[:, newaxis]**arange(N)
    B = A[0:n,0:m]
    return B

def my_regresion (x_datos, y_datos, degree):
  
    if ndim(x_datos) != 1 or ndim(y_datos) != 1 or shape(x_datos) != shape(y_datos):
        raise ValueError("Los vectores de datos no cuadran o no estan bien introducidos")
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("El grado debe ser un entero positivo")
    if degree >= len(x_datos):
        raise ValueError("El grado debe ser menor que el número de datos")

    V = my_Vandermonde(len(x_datos), degree + 1, x_datos)
    normal_matrix = matmul(transpose(V), V)
    right_hand_side = matmul(transpose(V), y_datos)

    return my_Gauss(normal_matrix, right_hand_side)

def my_Lagrange_pol(x_punto, x_datos, j):
    x_punto = np.asarray(x_punto, dtype=float)
    x_datos = np.asarray(x_datos, dtype=float)
    numerador = prod(x_punto - x_datos[i] for i in range(len(x_datos)) if i != j)
    denominador = prod(x_datos[j] - x_datos[i] for i in range(len(x_datos)) if i != j)
    return numerador / denominador

def my_interpolacion (x_punto, x_datos,y_datos,degree):
 return sum (y_datos[j] * my_Lagrange_pol(x_punto, x_datos, j) for j in range(degree + 1))

def temperatura_cohete(t):
    h_cohete = 300 * t
    h = np.clip(h_cohete, h_atm[0], h_atm[-1])
    T = zeros(len(h))

    for k, h_k in enumerate(h):
        j = np.searchsorted(h_atm, h_k, side="right") - 1
        j = np.clip(j, 0, len(h_atm) - 2)

        T[k] = my_interpolacion(h_k, h_atm[j:j + 2], T_atm[j:j + 2], 1)

    return T

def my_derivada_finita_1d (x_datos, y_datos):
    dx = x_datos[1] - x_datos[0]
    derivadas = zeros(len(x_datos)-1)
    for i in range (len(x_datos)-1):
        derivadas [i] = (y_datos[i+1]-y_datos[i])/dx
    return derivadas 

def my_derivada_finita_2d(x_datos, y_datos):
    n = len(x_datos)
    dx = x_datos[1] - x_datos[0]
    derivadas = zeros(n)
    if n>3:
        derivadas[0] = (-3*y_datos[0] + 4*y_datos[1] - y_datos[2]) / (2*dx)
        derivadas[-1] = (3*y_datos[-1] - 4*y_datos[-2] + y_datos[-3]) / (2*dx)

        for i in range(1, n - 1):
            derivadas[i] = (y_datos[i + 1] - y_datos[i - 1]) / (2*dx)
    else:
        raise ValueError("No hay suficientes datos para una derivada finita 2D")
    return derivadas

def my_integral_Riemann (x_datos, y_datos):
    dx = x_datos[1] - x_datos[0]

    S = sum (dx * y_datos[i] for i in range ((len(y_datos)-1)))

    return S

def my_integral_trapecios (x_datos, y_datos):
     dx = x_datos[1] - x_datos[0]
     n = len(y_datos)

     extremos = y_datos[0]+y_datos[-1]
     S = sum (2*y_datos[i] for i in range (1,n-1))

     return (dx/2)*(S + extremos)

def my_integral_Simpson (x_datos, y_datos):
     dx = x_datos[1] - x_datos[0]
     n = len(y_datos)

     if n%2 == 1: 
    
        extremos = y_datos[0]+y_datos[-1]
        S1=sum(4*y_datos[i] for i in range (1, n-1, 2))
        S2=sum(2*y_datos[i] for i in range (2, n-1, 2))
        
        return (dx/3)*(extremos+S1+S2)
     
     else:
         raise ValueError("Introduce un numero impar de datos")
