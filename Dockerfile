# Imagen base oficial de Python (slim para menor tamaño)
FROM python:3.11-slim

# Evitar prompts interactivos durante la instalación de paquetes
ENV DEBIAN_FRONTEND=noninteractive

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar primero el archivo de dependencias para aprovechar la caché de Docker
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Comando por defecto al iniciar el contenedor
CMD ["python", "main.py"]
