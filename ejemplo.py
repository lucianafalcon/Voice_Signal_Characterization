import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
Vin = 100e-3  # 100mV en voltios
k = 0.55e-3  # 0.55mA/V^2
VDD = 5  # 5V
RL = 16  # 16 ohm
rsig = 2200  # 2200 ohm
vsig = 100e-3  # 100mV en voltios
VT = 1.6  # 1.6V

# Funciones auxiliares
def gm(ID):
    return np.sqrt(2 * k * ID)

def Av0(gm, RD):
    return -gm * RD

def Avs(gm, RD, RL, Rin, rsig):
    return -gm * (RD * RL) / (RD + RL) * (Rin / (Rin + rsig))

def Rin(RG1, RG2):
    return RG1 * RG2 / (RG1 + RG2)

def VDS(ID, RD):
    return VDD - ID * RD

def vds(gm, vsig, RD):
    return gm * vsig * RD

# Algoritmo
def amplificador():
    # Paso 1: Elegir ID (Corriente de drenaje)
    I_D_values = np.linspace(1e-3, 100e-3, 100)  # Corrientes de drenaje de 0 a 100 mA

    Av_values = []  # Para almacenar las ganancias
    vgs_values = []  # Para almacenar los VGS - VT
    RD_optimal_values = []  # Para almacenar los RD óptimos
    RG1_values = []  # Para almacenar los valores de RG1
    RG2_values = []  # Para almacenar los valores de RG2
    max_gain = -np.inf  # Inicializar la ganancia máxima
    best_ID = None  # Mejor ID
    best_RD = None  # Mejor RD

    for ID in I_D_values:
        # Paso 3: Calcular VGS - VT
        VGS_minus_VT = np.sqrt(ID / k)

        # Paso 4: Calcular vgs-pico
        vgs_pico = max(min(0.2 * VGS_minus_VT, 100e-3), 0)
        vgs_values.append(vgs_pico)

        # Paso 5: Elegir RD y verificar distorsión
        RD_values = np.linspace(1, 1000, 100)  # Rango de RD para ajustar
        optimal_RD = None
        for RD in RD_values:
            # Verificar distorsión
            V_DS = VDS(ID, RD)
            vds_value = vds(gm(ID), vsig, RD)
            if V_DS + vds_value < VDD:
                # Hay distorsión por corte
                continue
            if vds_value > VDS(ID, RD) - VDS(ID, RD):
                # Hay distorsión por triodo
                continue
            optimal_RD = RD
            break

        if optimal_RD is None:
            optimal_RD = 1000  # Si no se encuentra un RD sin distorsión, elegir un valor grande

        RD_optimal_values.append(optimal_RD)

        # Calcular ganancia Av
        gm_value = gm(ID)
        Av = Av0(gm_value, optimal_RD)  # Ganancia sin considerar la carga de RL
        Av_values.append(Av)

        # Verificar si esta ganancia es la máxima
        if Av > max_gain:
            max_gain = Av
            best_ID = ID
            best_RD = optimal_RD

        # Calcular RG1 y RG2 (resistores de polarización)
        RG1 = 10e3  # Valor arbitrario de RG1
        RG2 = 10e3  # Valor arbitrario de RG2
        RG1_values.append(RG1)
        RG2_values.append(RG2)

    # Paso 6: Calcular Rin
    Rin_value = Rin(RG1, RG2)

    # Paso 7: Graficar Av vs ID
    plt.plot(I_D_values * 1e3, Av_values)  # Graficar en mA
    plt.xlabel('Corriente de drenaje (mA)')
    plt.ylabel('Ganancia (Av)')
    plt.title('Ganancia vs Corriente de drenaje')
    plt.grid(True)
    plt.show()

    # Imprimir algunos valores de RD, RG1, RG2, y ID
    for i in range(0, len(I_D_values), 10):  # Mostrar algunos valores a intervalos
        print(f"ID: {I_D_values[i]*1e3:.2f} mA, RD: {RD_optimal_values[i]:.2f} Ohm, RG1: {RG1_values[i]:.2f} Ohm, RG2: {RG2_values[i]:.2f} Ohm")

    # Imprimir el ID y RD que maximizan la ganancia
    print(f"\nEl valor de ID y RD con la máxima ganancia y sin distorsión son:")
    print(f"ID: {best_ID*1e3:.2f} mA, RD: {best_RD:.2f} Ohm")

    return Av_values, vgs_values, RD_optimal_values, RG1_values, RG2_values, best_ID, best_RD

# Ejecutar el algoritmo
Av_values, vgs_values, RD_optimal_values, RG1_values, RG2_values, best_ID, best_RD = amplificador()
