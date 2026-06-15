import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import spsolve

NUM_NODOS = 20
INCOGNITAS = NUM_NODOS ** 2
ENERGIA_F = 10000
h = 1.0/21

def main():
    
    
    A, b = calcular_Ab_eficiente()
    #plt.spy(A, markersize=1)

    #plt.title("Patrón de Bandas de la Matriz A")
    #plt.show()

    u = spsolve(A,b)
    u_internos = u.reshape((NUM_NODOS, NUM_NODOS))

    placa_completa = np.zeros((NUM_NODOS + 2, NUM_NODOS + 2))

    
    placa_completa[0, :] = 100  
    placa_completa[-1, :] = 0    
    placa_completa[:, 0] = 50   
    placa_completa[:, -1] = 50   

    placa_completa[1:-1, 1:-1] = u_internos


    colores = ["white", "yellow", "orange", "red"]
    
    mi_paleta = mcolors.LinearSegmentedColormap.from_list("mi_paleta", colores)

    x = np.linspace(0, 1, NUM_NODOS + 2)
    y = np.linspace(0, 1, NUM_NODOS + 2)
    X, Y = np.meshgrid(x, y)
   
    plt.figure(figsize=(10, 5))
    
    mapa = plt.contourf(X, Y, placa_completa, levels=20, cmap=mi_paleta, vmin=0, vmax=100)

    
    cbar = plt.colorbar(mapa)
    cbar.ax.set_title('°C', fontsize=10, pad=10, fontweight='bold')

    
    plt.title("Distribución de Temperatura con Fuente Central", fontweight='bold', pad=15)
    
    plt.gca().invert_yaxis() 

  
    plt.show()
    
def calcular_Ab_eficiente():
    arr_lat = np.full(INCOGNITAS - 1, -1.0)
    for i in range (NUM_NODOS - 1, INCOGNITAS - 1, NUM_NODOS):
        arr_lat[i] = 0.0

    diagonales = [-1, arr_lat, 4, arr_lat, -1]
    offsets =[-NUM_NODOS, -1, 0, 1, NUM_NODOS]

    A_sparse = sp.diags(diagonales, offsets, shape=(INCOGNITAS, INCOGNITAS), format='csr')

    b_2d = np.zeros((NUM_NODOS, NUM_NODOS))
    b_2d[0, :] += 100   
    b_2d[:, 0] += 50    
    b_2d[:, -1] += 50

    b_2d[9:11, 9:11] += (h ** 2) * (ENERGIA_F / 4)

    b = b_2d.flatten()

    return A_sparse, b

    

def calcular_Ab_ineficiente():
    A = np.zeros((INCOGNITAS, INCOGNITAS))
    b = np.zeros(INCOGNITAS)
    k = 0
    for i in range(0, NUM_NODOS):
        for j in range(0, NUM_NODOS):
            A[k][k] = 4

            if i-1 < 0:
                b[k] += 100
            else:
                A[k][k-NUM_NODOS] = -1

            if i+1 > 19:
                b[k] += 0
            else:
                A[k][k+NUM_NODOS] = -1
            
            if j-1 < 0:
                b[k] += 50
            else:
                A[k][k-1] = -1

            if j+1 > 19:
                b[k] += 50
            else:
                A[k][k+1] = -1
            
            if (i == 9 or i == 10) and (j == 9 or j == 10):
                b[k] += (h ** 2) * (ENERGIA_F/4)

            k += 1
        
    return A, b


if __name__ == "__main__":
    main()