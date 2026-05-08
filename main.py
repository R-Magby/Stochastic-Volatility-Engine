"""
Punto de entrada principal del proyecto de simulación Monte Carlo.
Reproduce el flujo completo del notebook MC_Heston_BSM.ipynb.
"""

from src import benchmark_simulaciones
import numpy as np

from src.logger import get_logger, log_parametros_simulacion, log_metricas_riesgo
from src.data_handler import descargar_datos, manipulate_data
from src.Model import BSMmodel, HestonModel 
from src.excel_reporter import generate_excel_report
from src.benchmark_simulaciones import BenchmarkSimulador

import time

logger = get_logger(__name__)

# Configuración
TICKER           = "SQM-B.SN"
FECHA_INICIO     = "2016-10-10"
N_SIMULACIONES   = 1_000
RANDOM_SEED      = 0


def main() -> None:
    """
    Orquesta el pipeline completo: descarga → análisis → simulación → reporte.
    """
    logger.info("======== Iniciando ========")
    # 1. Descarga de datos 
    logger.info("Descargando datos históricos desde Yahoo Finance...")
    try:
        df_sqm = descargar_datos(TICKER, start=FECHA_INICIO)
        logger.info("Datos descargados correctamente | filas=%d", len(df_sqm))
    except Exception as e:
        logger.critical("Fallo al descargar datos para %s: %s", TICKER, e, exc_info=True)
        raise

    # 2. Procesamiento y análisis exploratorio 
    logger.info("Procesando y analizando datos...")
    try:
        datos = manipulate_data(df_sqm)
        datos.informe_visual()

    except TypeError as e:
        logger.error("Error de tipo al procesar datos: %s", e, exc_info=True)
        raise
    except Exception as e:
        logger.critical("Error inesperado en procesamiento de datos: %s", e, exc_info=True)
        raise

    # 3. Parametros de simulacion
    np.random.seed(RANDOM_SEED)
    logger.debug("Semilla aleatoria fijada: %d", RANDOM_SEED)
    
    #busday excluye los fines de semana
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

    parametros = {"ticker":TICKER, "start_date":FECHA_INICIO, "end_date":datos.test.index[-1].date(),"simulation_date":datos.train.index[0].date(), "precio_inicial":datos.precio_inicial, "mu":datos.mu, "sigma":datos.sigma, "n_simulaciones":N_SIMULACIONES, "dias":n_dias}
    
    mensaje_parametros = "ticker={ticker}, start_date={start_date}, end_date={end_date}, simulation_date={simulation_date}, precio_inicial={precio_inicial}, mu={mu}, sigma={sigma}, n_simulaciones={n_simulaciones}, dias={dias}"
    log_parametros_simulacion(logger=logger, msg_parameter=mensaje_parametros, parametros=parametros)

    # 4. Benchmark de Simulaciones
    logger.info("Generando benchmark de simulaciones...")
    benchmark = BenchmarkSimulador(datos, dias_sim=n_dias, limite_combinaciones=50)
    
    # 1) Benchmark de BSM
    inicio = time.time()
    
    params_bsm = {
        "N": [10000],
        "optimize":["NumpyVectorization"]
    }
    try:
        benchmark.ejecutar_modelo(BSMmodel, "BSM", params_bsm)
    except Exception as e:
        logger.critical("Fallo durante la simulación Monte Carlo: %s", e, exc_info=True)
        raise

    # 2) Benchmark de Heston
    params_heston = {
        "N": [10000],
        "rho": [ 0.5, -0.5],
        "kappa": [0.01, 0.5],
        "theta": [0.04, 0.2],
        "v0": [None, 0.04],
        "optimize":['NumpyVectorization']
    }

    try:
        benchmark.ejecutar_modelo(HestonModel, "Heston", params_heston)
    except Exception as e:
        logger.critical("Fallo durante la simulación Monte Carlo: %s", e, exc_info=True)
        raise

    bench_result = benchmark.obtener_resultados_df()
    id_best, diff_best, best_model_dict  = benchmark.best_model()

    logger.info("Best model: %s", best_model_dict)   

    fin = time.time()
    logger.info(f"Duración Heston: {fin - inicio} segundos")
    # 5. Best Model

    try:   
        simulation = best_model_dict["func_model"](
            data_training=datos.train,
            data_test=datos.test,
            N_casos_posibles=N_SIMULACIONES,
            dias_de_simulacion=n_dias,
            **best_model_dict["parametros"]
        )

        simulation.simulate()
        logger.info("Simulacion Exitosa: %s", simulation.modelo)
    except Exception as e:
        logger.critical("Fallo durante la simulación Monte Carlo: %s", e, exc_info=True)
        raise

    # 6. Reporte y visualización 
    logger.info("Generando reportes y gráficos...")
    try:
        dicc_err, msj_log = simulation.analisis_de_errores()
        simulation.informe_visual()
        logger.info(msj_log)


        dicc_err.update({"modelo":simulation.modelo})  
        mensaje_riesgo ="Modelo {modelo} | VaR 95%: {VaR_95}%, CVaR 95%: {CVaR_95}%, Probabilidad de pérdida: {prob_loss}%, mse_price: {mse_price}, mae_price: {mae_price}"
        log_metricas_riesgo(logger, mensaje_riesgo, dicc_err)

        logger.info("Gráficos guardados correctamente.")
    except Exception as e:
        logger.error("Error al generar reportes: %s", e, exc_info=True)
        raise

    dict_simulation = simulation.paramteros_simulacion

    dummy_params = {"Id": id_best} | parametros | dict_simulation | best_model_dict
    
    generate_excel_report("reporte_simulacion_test.xlsx", dummy_params, dicc_err,bench_result)

 
    logger.info("Pipeline completado exitosamente.")

if __name__ == "__main__":
    main()
