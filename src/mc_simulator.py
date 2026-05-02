"""
Módulo del Simulador Monte Carlo (modelo Black-Scholes-Merton).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


class MonteCarloSimulator:
    """Simulador de Monte Carlo usando el modelo de Black-Scholes-Merton (BSM)."""

    def __init__(
        self,
        precio_inicial: float,
        mu: float,
        sigma: float = None,
        N_casos_posibles: int = 100,
        dias_de_simulacion: int = 31,
        modelo=None,
        log_retornos=None,
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
        self.mu = mu
        self.modelo = modelo
        self.sigma = sigma

        self.precio_inicial = precio_inicial
        self.dias_de_trending = dias_de_trending
        self.N_casos_posibles = N_casos_posibles
        self.dias_de_simulacion = dias_de_simulacion
        self.S_t = None

    def MC_simulation(self) -> np.ndarray:
        """
        Ejecuta la simulación de Monte Carlo (BSM) y devuelve las trayectorias.

        Input:
            None

        Output:
            np.ndarray: Matriz de shape (N_casos_posibles, dias_de_simulacion)
                        con los precios simulados.
        """
        dt = 1 / self.dias_de_trending
        Z = np.random.normal(size=(self.N_casos_posibles, self.dias_de_simulacion - 1))

        drift = self.mu - 0.5 * self.sigma ** 2
        self.S_t = np.ones((self.N_casos_posibles, self.dias_de_simulacion))

        self.S_t[:, :2] = self.precio_inicial

        self.S_t[:, 1:] = np.cumprod(
            self.S_t[:, 1:] * np.exp((drift * dt + self.sigma * Z * np.sqrt(dt))),
            axis=1,
        )

        return self.S_t

    def informe_visual(self):
        """
        Genera un gráfico con las trayectorias simuladas y la distribución del
        precio final.

        Input:
            None

        Output:
            None (muestra el gráfico)
        """
        fig, axl = plt.subplots(1, 2, figsize=(14, 6))

        for iter in range(self.N_casos_posibles):
            axl[0].plot(self.S_t[iter, :], alpha=0.05, color="grey")

        promedios = self.S_t.mean(axis=0)
        p25 = np.percentile(self.S_t, 25, axis=0)
        p50 = np.percentile(self.S_t, 50, axis=0)
        p75 = np.percentile(self.S_t, 75, axis=0)

        axl[0].fill_between(
            range(self.S_t.shape[1]),
            p25,
            p75,
            alpha=0.25,
            color='blue',
            label='IC 50%',
        )

        axl[0].fill_between(
            range(self.S_t.shape[1]),
            np.percentile(self.S_t, 5, axis=0),
            np.percentile(self.S_t, 95, axis=0),
            alpha=0.15,
            color='magenta',
            label='IC 90%',
        )

        axl[0].plot(promedios, color="red", label="Mean")

        perdida = (1 - promedios[-1] / self.precio_inicial) * 100
        signo = np.sign(promedios[-1] - self.precio_inicial)
        axl[0].set_title('Trayectorias simuladas')
        axl[0].set_xlabel("Dias")
        axl[0].set_ylabel("USD")
        axl[0].legend()

        N, bins, patches = axl[1].hist(
            self.S_t[:, -1], bins=40, edgecolor='none', alpha=0.7
        )
        fracs = N / N.max()
        norm = colors.Normalize(fracs.min(), fracs.max())
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)

        axl[1].axvline(x=p25[-1], color="r", linewidth=1.2, ls="--", label="Var 25%")
        axl[1].axvline(x=p75[-1], color="r", linewidth=1.2, ls="--")
        axl[1].axvline(self.precio_inicial, color='black', linestyle='--', label='Precio actual')
        axl[1].axvline(
            np.percentile(self.S_t[:, -1], 5), color='magenta', linestyle='--', label='VaR 95%'
        )

        axl[1].set_title('Distribución de precio final')
        axl[1].legend()

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
