# Como Ejecutar

## Requisitos
- [Docker Desktop](https://www.docker.com/products/docker-desktop) instalado

## Pasos

1. **Navega a la carpeta del proyecto:**
   ```bash
   cd ~/Desktop/ProyectoMarin
   ```

2. **Levanta los contenedores:**
   ```bash
   docker-compose up --build
   ```
   - Construye la imagen y ejecuta el contenedor
   - Lee el dataset etiquetado: `./data/jordiwild_preprocessed_v3 (1).csv`
   - Guarda la base de datos en: `./db_local/emociones_jordi.db`

3. **Listo** 
   - El script corre automaticamente
   - Los datos procesados estan en `db_local/`
   - Conecta la base de datos a Power BI para generar las graficas

## Para parar
```bash
docker-compose down
```

## Archivos del pipeline
- `src/limpieza.py` — Limpieza de texto (urls, menciones, emojis)
- `src/modelo_ia.py` — Modelo Naive Bayes + TF-IDF
- `src/pipeline.py` — Orquestador principal
