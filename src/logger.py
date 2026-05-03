"""
logger.py
=========

Genera tres archivos de log separados con rotación automática:
    - logs/info.log     → eventos operacionales normales (INFO y superiores)
    - logs/error.log    → errores recuperables (ERROR)
    - logs/critical.log → fallos no recuperables (CRITICAL)

Uso:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Mensaje informativo")
    logger.error("Algo salió mal")
    logger.critical("Fallo crítico del sistema")
"""

import logging
import logging.handlers
import os
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Configuración global
# ─────────────────────────────────────────────────────────────────────────────

# Directorio raíz del proyecto (un nivel arriba de src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directorio donde se almacenarán los logs
_LOG_DIR = _PROJECT_ROOT / "logs"

# Formato de los mensajes de log
_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Tamaño máximo por archivo antes de rotar (5 MB)
_MAX_BYTES = 5 * 1024 * 1024

# Número de archivos de backup a conservar
_BACKUP_COUNT = 3

# Flag para evitar inicializar handlers múltiples veces
_LOGGER_INITIALIZED = False


# Filtros personalizados

class _ExactLevelFilter(logging.Filter):
    """
    Filtra mensajes para que sólo pasen los de un nivel específico.
    """

    def __init__(self, level: int):
        super().__init__()
        self._level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == self._level


# Inicialización del sistema de logging

def _setup_logging() -> None:
    """
    Configura el sistema de logging con tres handlers de archivo más consola.
    Input:
        None

    Output:
        None
    """
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return

    # Crear directorio de logs si no existe
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Logger raíz del proyecto 
    root_logger = logging.getLogger("SimulationMonteCarlos")
    root_logger.setLevel(logging.DEBUG)

    # Evitar duplicación si el módulo se recarga
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Handler 1: info.log
    info_handler = logging.handlers.RotatingFileHandler(
        filename=_LOG_DIR / "info.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
        mode ="a"
    )
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    root_logger.addHandler(info_handler)

    # Handler 2: error.log 
    error_handler = logging.handlers.RotatingFileHandler(
        filename=_LOG_DIR / "error.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
        mode ="a"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.addFilter(_ExactLevelFilter(logging.ERROR))
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # ── Handler 3: critical.log  (sólo nivel CRITICAL) ───────────────────
    critical_handler = logging.handlers.RotatingFileHandler(
        filename=_LOG_DIR / "critical.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
        mode ="a"
    )
    critical_handler.setLevel(logging.CRITICAL)
    critical_handler.addFilter(_ExactLevelFilter(logging.CRITICAL))
    critical_handler.setFormatter(formatter)
    root_logger.addHandler(critical_handler)

    # ── Handler 4: consola  (DEBUG en desarrollo, INFO en producción) ─────
    console_handler = logging.StreamHandler()
    console_level = logging.DEBUG if os.getenv("MC_DEBUG", "0") == "1" else logging.INFO
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    _LOGGER_INITIALIZED = True
    root_logger.debug("Sistema de logging inicializado. Logs en: %s", _LOG_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Interfaz pública
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Devuelve un logger hijo del logger raíz del proyecto.
    Input:
        name (str): Nombre del módulo llamante, típicamente `__name__`.
    Output:
        logging.Logger: Logger listo para usar.
    """
    _setup_logging()
    if not name.startswith("SimulationMonteCarlos"):
        full_name = f"SimulationMonteCarlos.{name}"
    else:
        full_name = name
    return logging.getLogger(full_name)



# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de logging estructurado
# ─────────────────────────────────────────────────────────────────────────────

def log_parametros_simulacion(logger: logging.Logger,parameter : str) -> None:
    """
    Registra en INFO los parámetros clave de una simulación Monte Carlo.
    Input:
        logger           (logging.Logger): Logger del módulo llamante.
        ticker           (str):            Símbolo del activo.
        precio_inicial   (float):          Precio de entrada a la simulación.
        mu               (float):          Drift medio (retorno esperado diario).
        .                   .                           .
        .                   .                           .   

    Output:
        None
    """
    try:
        logger.info(parameter)
    except Exception as e:
        logger.error("Error al registrar parámetros de simulación: %s", e)


def log_metricas_riesgo(logger: logging.Logger, menssage_risk : str) -> None:
    """
    Registra en INFO las métricas de riesgo calculadas tras la simulación.

    Input:
        logger        (logging.Logger): Logger del módulo llamante.
        var_95        (float):          Value at Risk al 95% (como decimal).
        cvar_95       (float):          CVaR / Expected Shortfall al 95%.
        prob_perdida  (float):          Probabilidad de pérdida (0-1).
        prob_ganar_20 (float):          Probabilidad de ganancia > 20% (0-1).

    Output:
        None
    """
    logger.info(menssage_risk)
