"""
Módulo del Simulador Monte Carlo (modelo Black-Scholes-Merton).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import stats
from abc import ABC, abstractmethod
from src.logger import get_logger
logger = get_logger(__name__)


class MonteCarloSimulator:
    """Simulador de Monte Carlo usando el modelo de Black-Scholes-Merton (BSM)."""

    def __init__(
        self,
        data_training: np.ndarray,
        data_test: np.ndarray,
        N_casos_posibles: int = 100,
        dias_de_simulacion: int = 31,
        dias_de_trending: int = 252,
    ):
        """
        Inicializa el simulador con los parámetros del activo.

        Input:
            precio_inicial      (float): Precio inicial del activo.
            mu                  (float): Rendimiento medio (drift).
            sigma               (float): Volatilidad del activo.
            N_casos_posibles    (int):   Número de trayectorias a simular.
            dias_de_simulacion  (int):   Horizonte de simulación en días.
            modelo              (any):   Reservado para compatibilidad futura.
            log_retornos        (any):   Reservado para compatibilidad futura.
            dias_de_trending    (int):   Días de trading anuales (default 252).

        Output:
            None
        """

        self.data_training = data_training
        self.data_test = data_test
        self.rng = np.random.default_rng(seed=42)

        self.crecimiento =  np.log((self.data_training / self.data_training.shift(1)).dropna()).values

        self.mu = self.crecimiento.mean()
        self.sigma = self.crecimiento.std()

        self.precio_inicial = data_training.values[-1][0]

        self.dias_de_trending = dias_de_trending
        self.N_casos_posibles = N_casos_posibles
        self.dias_de_simulacion = dias_de_simulacion

        self.modelo = None
        self.S_t = None
        self.paramteros_simulacion={"mu":self.mu,"sigma":self.sigma,"precio_inicial":self.precio_inicial,"N_casos_posibles":self.N_casos_posibles,"dias_de_simulacion":self.dias_de_simulacion,"dias_de_trending":self.dias_de_trending}

    @abstractmethod
    def simulate(self) -> np.ndarray:
        """
        Ejecuta la simulación de Monte Carlo (BSM) y devuelve las trayectorias.
        """

    def informe_visual(self,):
        """
        Genera un gráfico con las trayectorias simuladas y la distribución del precio final.

        Input:
            None

        Output:
            None (muestra el gráfico)
        """
        if self.S_t is None:
            logger.error("Error en actualizacion de S_t")
            raise ValueError("Debe ejecutar el método simulate() antes de generar el informe visual.")
        
        p25 = np.percentile(self.S_t, 25, axis=0)
        p50 = np.percentile(self.S_t, 50, axis=0)
        p75 = np.percentile(self.S_t, 75, axis=0)
        promedios = self.S_t.mean(axis=0)

        fig, axl = plt.subplots(1, 3, figsize=(14, 6))

        # Grafico 1

        axl[0].plot(self.data_training, color = "blue",label="Data training",alpha=1.0)
        axl[0].plot(self.data_test, color = "green",label="Data test",alpha=1.0)

        #promedio de trayectorias
        axl[0].fill_between(
            self.data_test.reset_index().Date,
            np.percentile(self.S_t, 5, axis=0),
            np.percentile(self.S_t, 95, axis=0),
            alpha=0.15,
            color='magenta',
            label='IC 90%',
        )
        axl[0].fill_between(
            self.data_test.reset_index().Date,
            p25,
            p75,
            alpha=0.25,
            color='blue',
            label='IC 50%',
        )

        axl[0].plot([self.data_test.index[252].date(),self.data_test.index[252].date()], [0,self.data_test.max().values[0] * 1.5],"k--", label="Primer año",alpha=0.7, linewidth=0.8)

        axl[0].set_title('Prediccion vs Realidad')
        axl[0].set_xlabel("Dias")
        axl[0].tick_params(axis='x', rotation=45)
        axl[0].set_ylabel("USD")
        axl[0].set_ylim(self.data_test.min().values[0] * 0.9,self.data_test.max().values[0] * 1.5)
        try:
            axl[0].set_xlim(self.data_training.index[-252*3].date() , self.data_test.index[-1].date())
        except:
            axl[0].set_xlim(self.data_test.index[0].date() , self.data_test.index[-1].date())

        axl[0].legend()

        # Grafico 2
        for iter in range(self.N_casos_posibles):
            axl[1].plot(self.S_t[iter, :], alpha=0.05, color="grey")


        axl[1].fill_between(
            range(self.S_t.shape[1]),
            p25,
            p75,
            alpha=0.25,
            color='blue',
            label='IC 50%',
        )

        axl[1].fill_between(
            range(self.S_t.shape[1]),
            np.percentile(self.S_t, 5, axis=0),
            np.percentile(self.S_t, 95, axis=0),
            alpha=0.15,
            color='magenta',
            label='IC 90%',
        )

        axl[1].plot(p50, color="red", label="Median")

        perdida = (1 - p50[-1] / self.precio_inicial) * 100
        signo = np.sign(p50[-1] - self.precio_inicial)
        axl[1].set_title('Trayectorias simuladas')
        axl[1].set_xlabel("Dias")
        axl[1].set_ylabel("USD")
        axl[1].legend()

        N, bins, patches = axl[2].hist(
            self.S_t[:, -1], bins=40, edgecolor='none', alpha=0.7
        )
        fracs = N / N.max()
        norm = colors.Normalize(fracs.min(), fracs.max())
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)

        axl[2].axvline(x=p25[-1], color="r", linewidth=1.2, ls="--", label="Var 25%")
        axl[2].axvline(x=p75[-1], color="r", linewidth=1.2, ls="--")
        axl[2].axvline(self.precio_inicial, color='black', linestyle='--', label='Precio inicial')
        axl[2].axvline(np.percentile(self.S_t[:, -1], 5), color='magenta', linestyle='--', label='VaR 95%')

        axl[2].set_title('Distribución de precio final')
        axl[2].legend()

        fig.suptitle(
            f"Reporte de Rendimiento\n"
            f"Porcentaje de pérdida/ganancia: {signo * round(perdida, 2)}%",
            fontsize=12,
            color='black',
        )
        plt.savefig(f"Reporte Monte Carlo {self.modelo}.png")

        plt.close()
        

        
        plt.plot(self.data_training, color = "blue",label="Data training",alpha=1.0)
        plt.plot(self.data_test, color = "green",label="Data test",alpha=1.0)

        #promedio de trayectorias
        plt.fill_between(
            self.data_test.reset_index().Date,
            np.percentile(self.S_t, 5, axis=0),
            np.percentile(self.S_t, 95, axis=0),
            alpha=0.15,
            color='magenta',
            label='IC 90%',
        )
        plt.fill_between(
            self.data_test.reset_index().Date,
            p25,
            p75,
            alpha=0.25,
            color='blue',
            label='IC 50%',
        )

        plt.plot([self.data_test.index[252].date(),self.data_test.index[252].date()], [0,self.data_test.max().values[0] * 1.5],"k--", label="Primer año",alpha=0.7, linewidth=0.8)

        plt.title('Prediccion vs Realidad')
        plt.xlabel("Dias")
        plt.tick_params(axis='x', rotation=45)
        plt.ylabel("USD")
        plt.ylim(self.data_test.min().values[0] * 0.9,self.data_test.max().values[0] * 1.5)
        try:
            plt.xlim(self.data_training.index[-252*3].date() , self.data_test.index[-1].date())
        except:
            plt.xlim(self.data_test.index[0].date() , self.data_test.index[-1].date())

        plt.legend()
        plt.savefig(f"Reporte Monte Carlo {self.modelo} Prediccion vs Realidad.png")
        plt.close()

    def analisis_de_errores(self) -> str:
        #Percentiles
        mediana = np.percentile(self.S_t, 50, axis=0)

        # errores ^2 dan mas peso a valores muy alejados
        error_relativo_precio_mse = np.mean((self.data_test.values - mediana)**2)
        
        # MAE
        error_relativo_precio_mae = np.mean(np.abs(self.data_test.values - mediana))

        #Crecimiento
        Crecimiento_test = np.log((self.data_test / self.data_test.shift(1)).dropna()).values
        Crecimiento_simulado_media = np.median(np.log((self.S_t[:,:-1]/ self.S_t[:,1:])),axis=0)

        error_relativo_crecimiento_mse = np.mean((Crecimiento_test - Crecimiento_simulado_media)**2)
        error_relativo_crecimiento_mae = np.mean(np.abs(Crecimiento_test - Crecimiento_simulado_media))

        #test de Kolmogorov-Smirnov
        d, p = stats.ks_2samp(self.data_test.values.flatten(), mediana.flatten())
        ks_test = f": p={float(p):.3f}"
        if p < 0.05:
            ks_test += " | No se puede aceptar la hipótesis nula"
        else:
            ks_test += " | Se puede aceptar la hipótesis nula"

        # Intervalos de confianza
        ic_90 = np.percentile(self.S_t, [5, 95], axis=0)
        lower = ic_90[0]
        upper = ic_90[1]
        real = self.data_test.values

        coverage = np.mean((real >= lower) & (real <= upper))

        #percentage
        porcentajes_relative_price = mediana / self.data_test.values 

        precios_finales = self.S_t[:, -1]
        retornos = (precios_finales - self.precio_inicial) / self.precio_inicial

        VaR_95 = np.percentile(retornos, 5)
        CVaR_95 = retornos[retornos <= VaR_95].mean()
        prob_perdida = (precios_finales < self.precio_inicial).mean()
        prob_ganar_20 = (retornos > 0.20).mean()



        # Output
        dicc_errores = {
            "VaR_95": float(VaR_95),
            "CVaR_95": float(CVaR_95),
            "prob_loss": float(prob_perdida),
            "prob_gain_20": float(prob_ganar_20),
            "mse_price": float(error_relativo_precio_mse),
            "mae_price": float(error_relativo_precio_mae),
            "mse_growth": float(error_relativo_crecimiento_mse),
            "mae_growth": float(error_relativo_crecimiento_mae),
            "ks_test_d": d,
            "ks_test_p": float(p),
            "coverage": float(coverage),
            "porcentajes_relative_price": porcentajes_relative_price
        }
        mensaje_for_log = " VaR95 = {VaR_95},CVaR95={CVaR_95}, Probabilidad de perdida={prob_loss}, Probabilidad de ganancia del 20%={prob_gain_20}, Mse precio={mse_price}, Mae precio={mae_price}, Mse crecimiento={mse_growth}, Mae crecimiento={mae_growth}, Cobertura={coverage} | Test de Kolmogorov-Smirnov {ks_test_d}, {ks_test_p},porcentajes_relative_price {porcentajes_relative_price}"
        return dicc_errores, mensaje_for_log

        

