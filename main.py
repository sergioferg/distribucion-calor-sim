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
    
    
    A_con_fc, b_con_fc = calcular_Ab_eficiente(con_fuente=True)
    A_sin_fc, b_sin_fc = calcular_Ab_eficiente(con_fuente=False)

    u_con = spsolve(A_con_fc, b_con_fc).reshape((NUM_NODOS, NUM_NODOS))
    u_sin = spsolve(A_sin_fc, b_sin_fc).reshape((NUM_NODOS, NUM_NODOS))

    placa_con = construir_placa_completa(u_con)
    placa_sin = construir_placa_completa(u_sin)
    #plt.spy(A, markersize=1)

    #plt.title("Patrón de Bandas de la Matriz A")
    #plt.show()


    colores = ["white", "yellow", "orange", "red"]
    mi_paleta = mcolors.LinearSegmentedColormap.from_list("mi_paleta", colores)
    x = np.linspace(0, 1, NUM_NODOS + 2)
    y = np.linspace(0, 1, NUM_NODOS + 2)
    X, Y = np.meshgrid(x, y)

    # con fuente calor
    plt.figure(figsize=(10, 5))
    mapa_con = plt.contourf(X, Y, placa_con, levels=20, cmap=mi_paleta, vmin=0, vmax=100)
    cbar_con= plt.colorbar(mapa_con)
    cbar_con.ax.set_title('°C', fontsize=10, pad=10, fontweight='bold')
    plt.title("Distribución de Temperatura con Fuente Central", fontweight='bold', pad=15)
    plt.gca().invert_yaxis()
    plt.show()
    #sin fuente de calor
    plt.figure(figsize=(10, 5))
    mapa_sin = plt.contourf(X, Y, placa_sin, levels=20, cmap=mi_paleta, vmin=0, vmax=100)
    cbar = plt.colorbar(mapa_sin)
    cbar.ax.set_title('°C', fontsize=10, pad=10, fontweight='bold')
    plt.title("Distribución de Temperatura sin Fuente Central", fontweight='bold', pad=15)
    plt.gca().invert_yaxis()
    plt.show()

    #  u(x, 0.5)
    plt.figure(figsize=(10, 5))
    indice_centro = (NUM_NODOS + 2) // 2
    perfil_transversal = placa_con[indice_centro, :]
    plt.plot(x, perfil_transversal, color='red', marker='o', linestyle='-', linewidth=2)
    plt.title("Perfil Transversal de Temperatura para y = 0.5", fontweight='bold')
    plt.xlabel("Posición x")
    plt.ylabel("Temperatura (°C)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 110)
    plt.show()

def construir_placa_completa(u_internos):
    placa = np.zeros((NUM_NODOS + 2, NUM_NODOS + 2))
    placa[0, :] = 100  
    placa[-1, :] = 0    
    placa[:, 0] = 50   
    placa[:, -1] = 50   
    placa[1:-1, 1:-1] = u_internos
    return placa
    
def calcular_Ab_eficiente(con_fuente=True):
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

    if con_fuente:
        b_2d[9:11, 9:11] += (h ** 2) * (ENERGIA_F / 4)

    b = b_2d.flatten()

    return A_sparse, b

def calcular_Ab_ineficiente(con_fuente=True):
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

            if con_fuente and (i == 9 or i == 10) and (j == 9 or j == 10):
                b[k] += (h ** 2) * (ENERGIA_F/4)

            k += 1
        
    return A, b

if __name__ == "__main__":
    main()
