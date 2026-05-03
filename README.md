# 📈 Simulación Monte Carlo — Modelo Black-Scholes-Merton (BSM)

Proyecto de simulación estocástica de precios de activos financieros usando el
modelo de **Black-Scholes-Merton (BSM)** mediante el método de **Monte Carlo**.
El activo analizado es **SQM-B.SN** (SQM serie B, bolsa de Santiago), con datos
históricos descargados directamente desde Yahoo Finance.

---

## 🛠️ Herramientas utilizadas

| Herramienta | Versión recomendada | Uso |
|---|---|---|
| **Python** | 3.11 | Lenguaje principal |
| **yfinance** | ≥ 0.2 | Descarga de datos históricos |
| **NumPy** | ≥ 1.24 | Cálculo numérico y simulación |
| **Pandas** | ≥ 2.0 | Manejo de series temporales |
| **SciPy** | ≥ 1.11 | Estadísticos (skewness, kurtosis) |
| **Matplotlib** | ≥ 3.7 | Visualización de trayectorias |
| **Docker** | ≥ 24 | Entorno reproducible (sin instalar dependencias localmente) |

---

## 📐 Modelo Black-Scholes-Merton (BSM)

### Descripción

El modelo BSM supone que el precio de un activo $S_t$ sigue un **Movimiento
Geométrico Browniano (MGB)**, donde la volatilidad $\sigma$ es constante y los
log-retornos son independientes e idénticamente distribuidos con distribución
normal.

### Ecuación diferencial estocástica (SDE)

$$
dS_t = \mu \, S_t \, dt + \sigma \, S_t \, dW_t
$$

donde:

| Símbolo | Significado |
|---|---|
| $S_t$ | Precio del activo en el tiempo $t$ |
| $\mu$ | Drift (rendimiento medio continuo) |
| $\sigma$ | Volatilidad (desviación estándar de los log-retornos) |
| $dW_t$ | Incremento de un proceso de Wiener estándar |

### Solución cerrada (discretización)

Aplicando el lema de Itô, la solución exacta en pasos discretos de tamaño
$\Delta t = 1/252$ es:

$$
S_{t+\Delta t} = S_t \cdot \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t}\, Z\right], \quad Z \sim \mathcal{N}(0,1)
$$

### Métricas de riesgo calculadas

- **VaR 95 %** — Pérdida máxima esperada con 95 % de confianza.
- **CVaR 95 % (Expected Shortfall)** — Pérdida media en el peor 5 % de los escenarios.
- **Probabilidad de pérdida** — Fracción de trayectorias que terminan por debajo del precio inicial.
- **Probabilidad de ganancia > 20 %** — Fracción de trayectorias con retorno superior al 20 %.

---

## 📁 Estructura del proyecto

```
Heston y BSM/
├── src/
│   ├── __init__.py          # Exporta las clases principales
│   ├── data_handler.py      # Descarga y análisis exploratorio de datos
│   └── mc_simulator.py      # Clase MonteCarloSimulator (BSM)
├── main.py                  # Punto de entrada: orquesta el flujo completo
├── requirements.txt         # Dependencias de Python
├── Dockerfile               # Imagen Python 3.11-slim
└── docker-compose.yml       # Servicio Docker con volumen montado
```

---

## 🐳 Ejecución con Docker

> **Requisito previo:** tener [Docker Desktop](https://www.docker.com/products/docker-desktop/)
> instalado y en ejecución.

### 1. Clonar / posicionarse en el directorio del proyecto

```bash
cd "Stochastic-Volatility-Engine"
```

### 2. Construir la imagen

```bash
docker-compose build
```

Esto instala automáticamente todas las dependencias (incluyendo `yfinance`)
dentro del contenedor, sin necesidad de instalar nada en tu máquina local.

### 3. Ejecutar la simulación

```bash
docker-compose up
```

El contenedor descargará los datos de Yahoo Finance, calculará las estadísticas
del activo y guardará los gráficos como archivos PNG en el directorio actual:

- `Reporte Visual del Activo.png` — Serie histórica y distribución de precios.
- `Reporte Monte Carlo.png` — Trayectorias simuladas y distribución del precio final.

### 4. (Opcional) Abrir una shell interactiva dentro del contenedor

```bash
docker-compose run mc-simulator bash
```

Desde allí puedes ejecutar scripts de forma manual:

```bash
python main.py
```

### 5. Detener el contenedor

```bash
docker-compose down
```

---


### Reporte Visual del Activo
> Serie histórica (train/test) y distribución de precios del activo.

![Reporte Visual del Activo](Reporte%20Visual%20del%20Activo.png)

---

### Reporte Monte Carlo (BSM)
> Trayectorias simuladas con intervalos de confianza y distribución del precio final.

![Reporte Monte Carlo](Reporte%20Monte%20Carlo.png)


