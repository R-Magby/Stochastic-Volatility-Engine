"""
Punto de entrada principal del proyecto de simulación Monte Carlo.
Reproduce el flujo completo del notebook MC_Heston_BSM.ipynb.
"""

import numpy as np

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


def main() -> None:
    """
    Orquesta el pipeline completo: descarga → análisis → simulación → reporte.
    """
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
        datos.informe()
        datos.informe_visual()

    except TypeError as e:
        logger.error("Error de tipo al procesar datos: %s", e, exc_info=True)
        raise
    except Exception as e:
        logger.critical("Error inesperado en procesamiento de datos: %s", e, exc_info=True)
        raise

    # 3. Simulación Monte Carlo (BSM) 
    logger.info("Configurando simulación Monte Carlo (BSM)...")
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

    parametros = f"ticker={TICKER}, precio_inicial={datos.precio_inicial}, mu={datos.mu}, sigma={datos.sigma}, n_simulaciones={N_SIMULACIONES}, dias={n_dias}"
    log_parametros_simulacion(logger=logger, parameter=parametros)

    inicio = time.time()

    try:
        simulation = BSMmodel(
            data_training=datos.train,
            data_test=datos.test,
            N_casos_posibles=N_SIMULACIONES,
            dias_de_simulacion=n_dias,
            optimize="NumpyVectorization"
        )

        simulation.simulate()
        datos_simulados = simulation.S_t

    except Exception as e:
        logger.critical("Fallo durante la simulación Monte Carlo: %s", e, exc_info=True)
        raise

    fin = time.time()
    logger.info(f"Duración: {fin - inicio} segundos")

    # 4. Reporte y visualización 
    logger.info("Generando reportes y gráficos...")
    try:
        simulation.reporte()
        simulation.informe_visual()
        
        # Registrar métricas de riesgo en el log
        precios_finales = datos_simulados[:, -1]
        retornos = (precios_finales - datos.precio_inicial) / datos.precio_inicial
        var_95 = float(np.percentile(retornos, 5))
        cvar_95 = float(retornos[retornos <= var_95].mean())
        prob_perdida = float((precios_finales < datos.precio_inicial).mean())
        prob_ganar_20 = float((retornos > 0.20).mean())

        mensaje_riesgo = f"VaR 95%: {var_95 * 100:.2f}%, CVaR 95%: {cvar_95 * 100:.2f}%, Probabilidad de pérdida: {prob_perdida * 100:.1f}%, Probabilidad de +20%: {prob_ganar_20 * 100:.1f}%"
        log_metricas_riesgo(logger, mensaje_riesgo)

        logger.info("Gráficos guardados correctamente.")
    except Exception as e:
        logger.error("Error al generar reportes: %s", e, exc_info=True)
        raise

    logger.info("Pipeline completado exitosamente.")


if __name__ == "__main__":
    main()
