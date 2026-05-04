"""
Módulo de modelos de simulación Monte Carlo.
================================================================================
Contiene las implementaciones de los modelos:

    - BSMmodel      : Black-Scholes-Merton con drift constante.
    - HestonModel   : Modelo de volatilidad estocástica de Heston (1993).

Ambas clases heredan de `MonteCarloSimulator` (mc_simulator.py) e implementan
el método abstracto `simulate()`, que delega en dos estrategias de cómputo:
    - "For"                : ciclo explícito paso a paso (más legible).
    - "NumpyVectorization" : operaciones matriciales (más rápido).
"""

import numpy as np
from src.mc_simulator import MonteCarloSimulator
from src.logger import get_logger

logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Modelo Black-Scholes-Merton (BSM)
# ══════════════════════════════════════════════════════════════════════════════

class BSMmodel(MonteCarloSimulator):
    """
    Simulador Monte Carlo bajo el modelo Black-Scholes-Merton.

    Implementa la ecuación geométrica del movimiento browniano:
        S(t) = S(t-1) * exp((mu - 0.5*sigma²)*dt + sigma*sqrt(dt)*Z)

    Soporta dos estrategias de cómputo controladas por el parámetro
    `optimize`: "For" o "NumpyVectorization"

    Parameters:
    ----------
    precio_inicial :Precio de entrada del activo (S0).
    mu : Rendimiento medio diario (drift).
    sigma : Volatilidad histórica del activo.
    N_casos_posibles : Número de trayectorias a simular.
    dias_de_simulacion : Horizonte temporal en días.
    dias_de_trending : Días de trading anuales (default 252).
    optimize : Estrategia de cómputo: "For" | "NumpyVectorization".
    """
    def __init__(
        self,
        data_training: np.ndarray,
        data_test: np.ndarray,
        N_casos_posibles: int = 100,
        dias_de_simulacion: int = 31,
        dias_de_trending: int = 252,
        optimize: str = None
    ):
        super().__init__(
            data_training,
            data_test,
            N_casos_posibles,
            dias_de_simulacion,
            dias_de_trending,
        )
        try:
            self.optimize = optimize
            self.dt = 1 / dias_de_trending
            self.Z = np.random.normal(size=(N_casos_posibles, dias_de_simulacion - 1))
            self.drift = self.mu - 0.5 * self.sigma ** 2

            self.S_t = np.ones((N_casos_posibles, dias_de_simulacion))
            self.S_t[:, 0] = self.precio_inicial


            if self.optimize == "For":
                logger.info("Estrategia: ciclo For")
            elif self.optimize == "NumpyVectorization":
                logger.info("Estrategia: vectorización NumPy")
            else:
                logger.error(
                    "optimize='%s' no reconocido. "
                    "Usa 'For' o 'NumpyVectorization'.",
                    self.optimize
                )
        except Exception as exc:
            logger.critical("error en __init__: %s", exc, exc_info=True)
            raise

    def simulate(self):
        """
        Despacha la simulación BSM a la estrategia seleccionada.

        Returns
        -------
        None (actualiza self.S_t)

        Raises
        ------
        ValueError
            Si `optimize` no es un valor reconocido.
        """
        logger.info(
            "BSMmodel.simulate() | N=%d | T=%d | optimize=%s",
            self.N_casos_posibles, self.dias_de_simulacion, self.optimize,
        )

        if self.optimize == "For":
            for i in range(1, self.dias_de_simulacion):
                self.S_t[:, i] = self.S_t[:, i-1] * np.exp(self.drift * self.dt + self.sigma * self.Z[:, i-1] * np.sqrt(self.dt))

        elif self.optimize == "NumpyVectorization":
            self.S_t[:, 1]  = self.S_t[:, 0]
            self.S_t[:, 1:] = np.cumprod(
                self.S_t[:, 1:] * np.exp((self.drift * self.dt + self.sigma * self.Z * np.sqrt(self.dt))),
                axis=1,
            )

        else:
            msg = f"optimize='{self.optimize}' no reconocido. Usa 'For' o 'NumpyVectorization'."
            logger.error("BSMmodel | %s", msg)
            raise ValueError(msg)




# ══════════════════════════════════════════════════════════════════════════════
# Modelo de Heston (volatilidad estocástica)
# ══════════════════════════════════════════════════════════════════════════════

class HestonModel(MonteCarloSimulator):
    """
    Simulador Monte Carlo bajo el modelo de Heston (1993).

    El precio sigue un GBM con varianza estocástica que revierte a la media:
        dS = mu*S*dt + sqrt(V)*S*dW_S
        dV = kappa*(theta - V)*dt + sigma*sqrt(V)*dW_V
        corr(dW_S, dW_V) = rho

    Soporta dos estrategias: "For" (paso a paso) y "NumpyVectorization"
    (la evolución de la varianza es inherentemente secuencial; la parte
    vectorizada paraleliza el cómputo de S_t).
    """

    def __init__(
        self,
        data_training: np.ndarray,
        data_test: np.ndarray,
        N_casos_posibles: int = 100,
        dias_de_simulacion: int = 31,
        dias_de_trending: int = 252,
        rho: float = -0.7,
        kappa: float = 2.0,
        theta: float = 0.04,
        v0: float = None,
        optimize: str = None,
    ):
        super().__init__(
            data_training,
            data_test,
            N_casos_posibles,
            dias_de_simulacion,
            dias_de_trending,
        )
        try:
            self.optimize = optimize
            self.rho = rho
            self.kappa = kappa
            self.theta = theta
            self.v0 = self.sigma ** 2 if v0 is None else v0
            self.dt = 1 / self.dias_de_trending

            if self.optimize == "For":
                logger.info("Estrategia: ciclo For")
            elif self.optimize == "NumpyVectorization":
                logger.info("Estrategia: vectorización NumPy")
            else:
                msg = (
                    f"optimize='{self.optimize}' no reconocido. "
                    "Usa 'For' o 'NumpyVectorization'."
                )
                logger.error("HestonModel | %s", msg)
                raise ValueError(msg)

        except Exception as exc:
            logger.critical("HestonModel | error en __init__: %s", exc, exc_info=True)
            raise

    # ── Inicialización de variables estocásticas compartidas ──────────────

    def _init_variables(self) -> None:
        """
        Genera los ruidos brownianos correlacionados y las matrices de
        trayectorias para precio (S_t) y varianza (V_t).

        Se llama internamente antes de ejecutar ForLoop o NumpyVectorization.
        """
        try:
            Z_price = np.random.normal(size=(self.N_casos_posibles, self.dias_de_simulacion - 1))
            Z_vol = np.random.normal(size=(self.N_casos_posibles, self.dias_de_simulacion - 1))
            Z_vol_corr = self.rho * Z_price + np.sqrt(1 - self.rho**2) * Z_vol

            self.Z_price = Z_price
            self.Z_vol = Z_vol_corr

            self.S_t = np.empty((self.N_casos_posibles, self.dias_de_simulacion))
            self.V_t = np.empty((self.N_casos_posibles, self.dias_de_simulacion))

            self.S_t[:, 0] = self.precio_inicial
            self.V_t[:, 0] = self.v0

            logger.debug(
                "HestonModel | variables estocásticas inicializadas | "
                "rho=%.2f | v0=%.6f | kappa=%.2f | theta=%.4f",
                self.rho, self.v0, self.kappa, self.theta,
            )
        except Exception as exc:
            logger.critical(
                "HestonModel | error al inicializar variables: %s", exc, exc_info=True
            )
            raise

    # ── Método abstracto implementado ─────────────────────────────────────

    def simulate(self) -> np.ndarray:
        """
        Despacha la simulación Heston a la estrategia seleccionada.

        Returns
        -------
        np.ndarray
            Matriz (N_casos_posibles, dias_de_simulacion) de precios simulados.
        """
        logger.info(
            "HestonModel.simulate() | N=%d | T=%d | optimize=%s",
            self.N_casos_posibles, self.dias_de_simulacion, self.optimize,
        )
        self._init_stochastic_vars()

        if self.optimize == "For":
            return self.ForLoop()
        elif self.optimize == "NumpyVectorization":
            return self.NumpyVectorization()
        else:
            msg = f"optimize='{self.optimize}' no reconocido."
            logger.error("HestonModel | %s", msg)
            raise ValueError(msg)

    # ── Estrategia 1: ciclo For explícito ────────────────────────────────

    def ForLoop(self) -> np.ndarray:
        """
        Simula precio y varianza paso a paso (método de Euler-Maruyama).

        Returns
        -------
        None (Actualiza self.S_t y self.V_t)
        """
        try:
            logger.info("HestonModel | ForLoop iniciado | T=%d pasos", self.dias_de_simulacion)
            sqrt_dt = np.sqrt(self.dt)

            for t in range(1, self.dias_de_simulacion):
                V_prev = self.V_t[:, t - 1]
                V_sqrt = np.sqrt(np.maximum(V_prev, 0))

                # Euler-Maruyama para la varianza (CIR process)
                dV = (self.kappa * (self.theta - V_prev) * self.dt + self.sigma * V_sqrt * self.Z_vol[:, t - 1] * sqrt_dt)
                self.V_t[:, t] = np.maximum(V_prev + dV, 1e-10)

                # GBM con varianza estocástica
                vol_sqrt = np.sqrt(self.V_t[:, t - 1])
                drift = self.mu - 0.5 * self.V_t[:, t - 1]
                dS = drift * self.dt + vol_sqrt * self.Z_price[:, t - 1] * sqrt_dt
                self.S_t[:, t] = self.S_t[:, t - 1] * np.exp(dS)

            logger.info("HestonModel | ForLoop completado.")
        except Exception as exc:
            logger.critical("HestonModel | ForLoop fallido: %s", exc, exc_info=True)
            raise

    # ── Estrategia 2: vectorización NumPy ────────────────────────────────

    def NumpyVectorization(self) -> np.ndarray:
        """
        Simula precio y varianza usando operaciones vectoriales de NumPy.
        Returns
        -------
        None (Actualiza self.S_t y self.V_t)
        """
        try:
            logger.info(
                "HestonModel | NumpyVectorization iniciado | T=%d pasos",
                self.dias_de_simulacion,
            )
            sqrt_dt = np.sqrt(self.dt)

            for t in range(1, self.dias_de_simulacion):
                V_prev = self.V_t[:, t - 1]
                V_sqrt = np.sqrt(np.maximum(V_prev, 0))

                # Varianza                
                dV = self.kappa * (self.theta - V_prev) * self.dt + self.sigma * V_sqrt * self.Z_vol[:, t-1]
                self.V_t[:, t] = V_prev + dV
                
                self.V_t[:, t] = np.maximum(self.V_t[:, t], 1e-10)

                # Precio 
                vol_sqrt = np.sqrt(self.V_t[:, t - 1])
                drift = self.mu - 0.5 * self.V_t[:, t - 1]
                dS = drift * self.dt + vol_sqrt * self.Z_price[:, t - 1] * sqrt_dt
                self.S_t[:, t] = self.S_t[:, t - 1] * np.exp(dS)

            logger.info("HestonModel | NumpyVectorization completado.")
        except Exception as exc:
            logger.critical(
                "HestonModel | NumpyVectorization fallido: %s", exc, exc_info=True
            )
            raise

        return self.S_t