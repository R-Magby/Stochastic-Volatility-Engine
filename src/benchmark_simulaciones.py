import itertools
import time
import numpy as np

from src.logger import get_logger
from src.data_handler import descargar_datos, manipulate_data
from src.Model import BSMmodel, HestonModel

logger = get_logger(__name__)

class BenchmarkSimulador:

    def __init__(self, datos, dias_sim, limite_combinaciones=50):
        """
        Inicializa el evaluador de modelos.
        
        Args:
            datos: Objeto con train y test data segmentados (manipulate_data).
            dias_sim: Días de simulación hacia el futuro.
            limite_combinaciones: Tope de combinaciones máximo para evitar desbordes de memoria o tiempo.
        """
        self.datos = datos
        self.dias_sim = dias_sim
        self.limite_combinaciones = limite_combinaciones
        self.resultados = []
        self.modelo_clase = None

    def ejecutar_modelo(self, modelo_clase, nombre_modelo, dict_listas_params):
        """
        Ejecuta todas las combinaciones posibles de parámetros dados en dict_listas_params.
        
        Args:
            modelo_clase: Clase del modelo a instanciar (BSMmodel o HestonModel).
            nombre_modelo: Nombre descriptivo para guardar en los resultados.
            dict_listas_params: Diccionario de listas con parámetros. Ejemplo:
                                {'N': [100, 1000], 'rho': [-0.5, 0.5]}
        """
        # Crear el producto cartesiano de todas las listas de hiperparámetros
        keys, values = zip(*dict_listas_params.items())
        combinaciones = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        if len(combinaciones) > self.limite_combinaciones:
            logger.warning(f"Se excede el limite de combinaciones ({len(combinaciones)} > {self.limite_combinaciones}). Truncando a las primeras {self.limite_combinaciones}.")
            combinaciones = combinaciones[:self.limite_combinaciones]
            
        #logger.info(f"Ejecutando benchmark para {nombre_modelo} con {len(combinaciones)} combinaciones.")

        self.modelo_clase = modelo_clase

        for i, params in enumerate(combinaciones):
            # Extraer N porque el modelo lo recibe como N_casos_posibles
            n_simulaciones = params.pop("N", 1000)
            try:
                t0 = time.perf_counter()
                
                # Instanciar el modelo pasando dinámicamente los parámetros usando **params
                modelo_inst = modelo_clase(
                    data_training=self.datos.train,
                    data_test=self.datos.test,
                    N_casos_posibles=n_simulaciones,
                    dias_de_simulacion=self.dias_sim,
                    **params
                )
                
                # Ejecutar simulación
                modelo_inst.simulate()
                duracion = time.perf_counter() - t0
                
                # Obtener análisis de error y sus respectivas métricas
                dicc_err, _ = modelo_inst.analisis_de_errores()
                
                # Restaurar N_simulaciones para guardarlo íntegro en la salida
                
                resultado = {"Id":f"{i+1}/{len(combinaciones)}",
                    "modelo": nombre_modelo,
                    "parametros": params.copy(),
                    "N_simulaciones": n_simulaciones,
                    "tiempo_ejecucion": duracion,
                    "MAE": dicc_err.get("mae_price", np.nan),
                    "MSE": dicc_err.get("mse_price", np.nan),
                    "Coverage": dicc_err.get("coverage", np.nan)
                }                
                self.resultados.append(resultado)
                #logger.info(f"[{i+1}/{len(combinaciones)}] OK - {nombre_modelo} (N={n_simulaciones}) | MAE: {resultado['MAE']:.2f} | MSE: {resultado['MSE']:.2f} | Coverage: {resultado['Coverage']:.2f} | Tiempo: {duracion:.2f}s")
                
            except Exception as e:
                logger.error(f"Error al evaluar {nombre_modelo} con params {params}: {e}", exc_info=True)

    def best_model(self):
        """
        Obtiene el mejor modelo.
        """
        id_best = None
        coverage_best = 0.90
        diff = np.inf
        best_model_dict = None
        for modelo in self.resultados:
            if abs(modelo["Coverage"] - coverage_best) < diff:
                diff = abs(modelo["Coverage"] - coverage_best)
                id_best = modelo["Id"]
                best_model_dict = modelo
        
        best_model_dict = {"func_model": self.modelo_clase } | best_model_dict
        return id_best, diff, best_model_dict 

    def obtener_resultados_df(self):
        """Devuelve un DataFrame resumen de todas las ejecuciones almacenadas."""
        return self.resultados



