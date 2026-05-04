"""
Módulo del Simulador Monte Carlo (modelo Black-Scholes-Merton).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
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
        modelo: str = None,
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

        self.crecimiento =  np.log((self.data_training / self.data_training.shift(1)).dropna()).values

        self.mu = self.crecimiento.mean()
        self.sigma = self.crecimiento.std()

        self.precio_inicial = data_training.values[-1][0]

        self.dias_de_trending = dias_de_trending
        self.N_casos_posibles = N_casos_posibles
        self.dias_de_simulacion = dias_de_simulacion

        self.modelo = modelo
        self.S_t = None

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
        axl[0].plot(self.data_test.reset_index().Date, promedios, color="red", label="Mean",alpha=0.7, linewidth=0.8)

        axl[0].set_title('Prediccion vs Realidad')
        axl[0].set_xlabel("Dias")
        axl[0].set_ylabel("USD")
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

        axl[1].plot(promedios, color="red", label="Mean")

        perdida = (1 - promedios[-1] / self.precio_inicial) * 100
        signo = np.sign(promedios[-1] - self.precio_inicial)
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
        axl[2].axvline(self.precio_inicial, color='black', linestyle='--', label='Precio actual')
        axl[2].axvline(np.percentile(self.S_t[:, -1], 5), color='magenta', linestyle='--', label='VaR 95%')

        axl[2].set_title('Distribución de precio final')
        axl[2].legend()

        fig.suptitle(
            f"Reporte de Rendimiento\n"
            f"Porcentaje de pérdida/ganancia: {signo * round(perdida, 2)}%",
            fontsize=12,
            color='black',
        )
        plt.savefig("Reporte Monte Carlo.png")

        #plt.show()

    def reporte(self):
        """
        Imprime métricas de riesgo: VaR, CVaR y probabilidades.
        Input:
            None

        Output:
            None (imprime por consola)
        """
        precios_finales = self.S_t[:, -1]
        retornos = (precios_finales - self.precio_inicial) / self.precio_inicial

        VaR_95 = np.percentile(retornos, 5)
        CVaR_95 = retornos[retornos <= VaR_95].mean()
        prob_perdida = (precios_finales < self.precio_inicial).mean()
        prob_ganar_20 = (retornos > 0.20).mean()

        print(f"Precio inicial:                 ${self.precio_inicial:.2f}")
        print(f"Precio medio final:             ${precios_finales.mean():.2f}")
        print(f"VaR 95%:                        {VaR_95 * 100:.2f}%")
        print(f"CVaR 95% (Expected Shortfall):  {CVaR_95 * 100:.2f}%")
        print(f"Probabilidad de pérdida:        {prob_perdida * 100:.1f}%")
        print(f"Probabilidad de +20%:           {prob_ganar_20 * 100:.1f}%")
