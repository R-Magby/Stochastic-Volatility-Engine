"""
Punto de entrada principal del proyecto de simulación Monte Carlo.
Reproduce el flujo completo del notebook MC_Heston_BSM.ipynb.
"""

import numpy as np
from src.data_handler import descargar_datos, manipulate_data
from src.mc_simulator import MonteCarloSimulator


def main():
    # ------------------------------------------------------------------ #
    # 1. Descarga de datos                                                 #
    # ------------------------------------------------------------------ #
    df_sqm = descargar_datos("SQM-B.SN", start="2016-10-10")

    # ------------------------------------------------------------------ #
    # 2. Procesamiento y análisis exploratorio                             #
    # ------------------------------------------------------------------ #
    datos = manipulate_data(df_sqm)
    datos.informe()
    datos.informe_visual()

    # ------------------------------------------------------------------ #
    # 3. Simulación Monte Carlo (BSM)                                      #
    # ------------------------------------------------------------------ #
    np.random.seed(0)

    simulation = MonteCarloSimulator(
        precio_inicial=datos.precio_inicial,
        mu=datos.mu,
        sigma=datos.sigma,
        N_casos_posibles=1000,
        dias_de_simulacion=int(datos.days / 8),
    )

    datos_simulados = simulation.MC_simulation()
    simulation.reporte()
    simulation.informe_visual()


if __name__ == "__main__":
    main()
