from src import benchmark_simulaciones
import numpy as np
import matplotlib.pyplot as plt

from src.logger import get_logger, log_parametros_simulacion, log_metricas_riesgo
from src.data_handler import descargar_datos, manipulate_data
from src.Model import BSMmodel, HestonModel 

import time

logger = get_logger(__name__)

# Configuración
TICKER           = "SQM-B.SN"
FECHA_INICIO     = "2016-10-10"
N_SIMULACIONES   = 1_000
RANDOM_SEED      = 0

def plotear(convergencia: list,N_list: list, name: str):
    Vector_per_N = []
    For_per_N = []
    Multi_1_per_N = []
    Multi_2_per_N = []


    for i,temp in enumerate(N_list):
        pack_convergencia = convergencia[i*4: (i+1)*4]
        For_per_N.append(pack_convergencia[0]["time"])
        Vector_per_N.append(pack_convergencia[1]["time"])
        Multi_1_per_N.append(pack_convergencia[2]["time"])
        if i == len(N_list)-1:
            pass
        else:
            Multi_2_per_N.append(pack_convergencia[3]["time"])

        
    try:
        plt.plot(N_list,Vector_per_N,label="Vectorrizacion Numpy")
        plt.plot(N_list,For_per_N,label="For")
        plt.plot(N_list,Multi_1_per_N,label="Multiprocessing 1 core")
        plt.plot(N_list[:-1],Multi_2_per_N,label="Multiprocessing 2 cores")

        plt.xscale("log", base=10)

        plt.xlabel("N: Número de simulaciones")
        plt.ylabel("Tiempo de ejecución (segundos)")

        plt.ylim(0, 15)
        plt.title(f"Comparación de tiempos de computo | {name}")
        plt.legend()
        plt.savefig(f"estudio_speed {name}.png")
        plt.close()
    except Exception as e:
        logger.critical("Fallo al generar el gráfico: %s", e, exc_info=True)
        raise   
        

def main() -> None:

    logger.info("Iniciando simulacion estudio de tiempo de computo...")
    
    # 1. Descarga de datos 
    logger.info(f"Descargando datos históricos desde Yahoo Finance para {TICKER}...")
    try:
        df_sqm = descargar_datos(TICKER, start=FECHA_INICIO)
        datos = manipulate_data(df_sqm)
        datos.segmentar_data()
        logger.info("Datos descargados correctamente | filas=%d", len(df_sqm))
    except Exception as e:
        logger.critical("Fallo al descargar datos para %s: %s", TICKER, e, exc_info=True)
        raise

    # 2. Parametros
    n_dias = np.busday_count(datos.test.index[0].date(), datos.test.index[-1].date())
    try:
        if n_dias ==datos.test.shape[0]:
            pass
        else:
            n_dias = datos.test.shape[0]
            logger.warning("El número de días no coincide con el número de días de test. Se utilizará el número de días de test. Es necesario el calendario financiero para obtener el número exacto de días.")
    except ValueError as e:
        logger.error("Error al procesar el calendario financiero: %s", e, exc_info=True)
        raise

    N_sim = [10, 50, 100, 250, 500, 750 , 1000, 2500, 5000, 7500, 10000, 25000, 50000]
    optimizacion = ["For","NumpyVectorization","Multiprocessing"]
    core_list = [1,2]
    """params_heston = {
        "rho": [ 0.5, -0.5],
        "kappa": [0.01, 0.5],
        "theta": [0.04, 0.2],
        "v0": [None, 0.04],}"""
    #optimizacion = ["NumpyVectorization", "PurePython", "Numba", "NumExpr"]

    # 3. Simulaciones

    convergencia_BSM = []
    convergencia_Heston = []

    logger.info("Parametros de simulacion: N_sim=%s, optimizacion=%s, core_list=%s", N_sim, optimizacion, core_list)
    Parametros_heston = []


    try:
        for model in [ BSMmodel, HestonModel]:
            for n in N_sim:
                for opt in optimizacion:
                    if model == BSMmodel:
                        params = {"optimize" : opt, "N_casos_posibles": n }
                    else:
                        params = {"optimize" : opt, "N_casos_posibles": n,"rho":0.7,"kappa":0.01,"theta":0.04,"v0":None }
                        
                    if opt == "Multiprocessing": 
                        for core in core_list:
                            start_tiempo = time.time()
                            if n == 50000 and core == 2:
                                pass
                            else:
                                simulation = model(data_training=datos.train, data_test=datos.test, dias_de_simulacion=n_dias,cores=core, **params)
                                simulation.simulate()
                                dicc_err, msj_log = simulation.analisis_de_errores()

                                end_tiempo = time.time()
                            if model == BSMmodel:
                                convergencia_BSM.append({"N_sim":n, "model":str(model), "optimize":opt, "cores":core, "coverage":dicc_err["coverage"], "MSE":dicc_err["mse_price"], "MAE":dicc_err["mae_price"], "time":end_tiempo-start_tiempo})
                            else:

                                convergencia_Heston.append({"N_sim":n, "model":str(model), "optimize":opt, "cores":core, "coverage":dicc_err["coverage"], "MSE":dicc_err["mse_price"], "MAE":dicc_err["mae_price"], "time":end_tiempo-start_tiempo})
                    else:
                        start_tiempo = time.time()

                        simulation = model(data_training=datos.train, data_test=datos.test, dias_de_simulacion=n_dias, **params)
                        simulation.simulate()
                        dicc_err, msj_log = simulation.analisis_de_errores()

                        end_tiempo = time.time()

                        if model == BSMmodel:
                            convergencia_BSM.append({"N_sim":n, "model":str(model), "optimize":opt, "cores":"---", "coverage":dicc_err["coverage"], "MSE":dicc_err["mse_price"], "MAE":dicc_err["mae_price"], "time":end_tiempo-start_tiempo})
                        else:
                            convergencia_Heston.append({"N_sim":n, "model":str(model), "optimize":opt, "cores":"---", "coverage":dicc_err["coverage"], "MSE":dicc_err["mse_price"], "MAE":dicc_err["mae_price"], "time":end_tiempo-start_tiempo})

        logger.info("Ciclo completado")
    except Exception as e:
        logger.critical("Fallo durante la simulación Monte Carlo BSM: %s", e, exc_info=True)
        raise

    
    with open("Reporte Historico.txt", "w") as archivo:
        for i,temp in enumerate(convergencia_BSM + convergencia_Heston):
            archivo.write(f"Id: {i} | modelo: {temp.get('model')} | N: {temp.get('N_sim')} | Optimizador: {temp.get('optimize')} | Cores: {temp.get('cores')} | Coverage: {temp.get('coverage')} | MSE: {temp.get('MSE')} | MAE: {temp.get('MAE')} | Time: {temp.get('time')}\n")
    
    
    plotear(convergencia_BSM,N_sim," BSM ")
    plotear(convergencia_Heston,N_sim," Heston ")

    logger.info("Estudio completado")
    logger.info("Resultados guardados en Reporte Historico.txt y estudio_speed.png")


if __name__ == "__main__":
    main()
