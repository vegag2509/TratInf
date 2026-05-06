FROM python:3.10-slim
WORKDIR /app
# Copia el archivo de requisitos desde la carpeta docker
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copia todo el contenido de ProyectoMarin al contenedor
COPY . .
# Ejecuta tu script de IA
# CMD ["python", "data/main.py"]
CMD ["python", "src/pipeline.py"]