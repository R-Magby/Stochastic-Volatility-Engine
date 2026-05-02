"""
Módulo de manejo y procesamiento de datos financieros.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from matplotlib import colors


def descargar_datos(ticker: str, start: str) -> pd.DataFrame:
    """
    Descarga datos históricos de un activo desde Yahoo Finance.

    Input:
        ticker (str): Símbolo del activo (ej. 'SQM-B.SN').
        start  (str): Fecha de inicio en formato 'YYYY-MM-DD'.

    Output:
        pd.DataFrame: DataFrame con precios históricos (OHLCV).
    """
    return yf.download(ticker, start=start)


class manipulate_data:
    """Clase para manipular y analizar datos financieros de un activo."""

    def __init__(self, activos: pd.DataFrame, trading_days: int = 252):
        """
        Inicializa la clase con un DataFrame de precios.

        Input:
            activos      (pd.DataFrame): DataFrame con columna 'Close'.
            trading_days (int):          Días de trading anuales (default 252).

        Output:
            None
        """
        if not isinstance(activos, pd.DataFrame):
            raise TypeError("Se espera un DataFrame.")

        self.activos = activos
        self.trading_days = trading_days

        self.crecimiento = np.log((self.activos['Close'] / self.activos['Close'].shift(1)).dropna()).values

        self.mu = self.crecimiento.mean()
        self.sigma = self.crecimiento.std()
        self.precio_inicial = activos["Close"].values[-1][0]

        self.days = np.abs((self.activos.index[0] - self.activos.index[-1]).days)

        self.dia_de_corte = None
        self.test = None
        self.train = None

    def informe(self):
        """
        Imprime un resumen estadístico del activo.
        Input:
            None
        Output:
            None (imprime por consola)
        """
        print("Dias pasados:", self.days)
        print("Precio actual:", self.precio_inicial)
        print("Volatilidad anual:", self.sigma)
        print("Riesgo anual:", self.mu)

    def segmentar_data(self):
        """
        Divide los datos en conjuntos de entrenamiento (80%) y prueba (20%).
        Input:
            None
        Output:
            None (actualiza self.train y self.test)
        """
        self.dia_de_corte = int(self.activos.shape[0] * 0.8)
        self.test = self.activos.Close.iloc[self.dia_de_corte:]
        self.train = self.activos.Close.iloc[:self.dia_de_corte]

    def informe_visual(self):
        """
        Genera un gráfico con la serie de precios (train/test) y su distribución.
        Input:
            None
        Output:
            gráficos
        """
        self.segmentar_data()

        fig, axl = plt.subplots(1, 2, figsize=(14, 6))

        axl[0].plot(self.test, alpha=0.5, color="red")
        axl[0].plot(self.train, alpha=0.85, color="blue")

        axl[0].set_title(f"Activos | dia de corte: {self.activos.reset_index().Date.dt.date[self.dia_de_corte]}")

        axl[0].set_xlabel("Dias")
        axl[0].set_ylabel("USD")
        axl[0].legend(["Test", "Train"])

        N, bins, patches = axl[1].hist(
            self.activos.Close, bins=40, edgecolor='none', alpha=0.7
        )
        fracs = N / N.max()
        norm = colors.Normalize(fracs.min(), fracs.max())
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)

        axl[1].set_title(f"Distribución | Asimetria: {skew(self.activos.Close).round(2)} | Curtosis {kurtosis(self.activos.Close).round(2)}" )
        axl[1].legend()

        fig.suptitle("Reporte Visual del Activo")
        plt.savefig("Reporte Visual del Activo.png")
        #plt.show()
