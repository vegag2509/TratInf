import os
import sys
import pandas as pd
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from limpieza import limpiar_texto
from modelo_ia import entrenar_y_predecir


def main():
    # Detectar entorno (Docker vs local)
    if os.path.isdir("/app/data"):
        ruta_csv = "/app/data/jordiwild_preprocessed_v3 (1).csv"
        ruta_db_dir = "/app/db"
    else:
        base = os.path.join(os.path.dirname(__file__), "..")
        ruta_csv = os.path.join(base, "data", "jordiwild_preprocessed_v3 (1).csv")
        ruta_db_dir = os.path.join(base, "db_local")

    ruta_db = os.path.join(ruta_db_dir, "emociones_jordi.db")
    ruta_reporte = os.path.join(ruta_db_dir, "reporte_modelo.txt")

    print(f"[PIPELINE] Leyendo CSV: {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    print(f"[PIPELINE] Registros leidos: {len(df)}")

    # Limpiar texto
    df["Tweet_Limpio"] = df["Tweet Content"].astype(str).apply(limpiar_texto)
    print("[PIPELINE] Limpieza de texto completada.")

    # Usar etiquetas reales del dataset
    print("[PIPELINE] Usando etiquetas 'Emocion' del archivo.")
    print("[PIPELINE] Distribucion de emociones:")
    print(df["Emocion"].value_counts())

    # Filtrar clases con muy pocos ejemplos (menos de 5)
    # El modelo no puede aprender de 1 o 2 tweets, asi que las eliminamos
    conteo = df["Emocion"].value_counts()
    clases_validas = conteo[conteo >= 5].index.tolist()
    clases_eliminadas = conteo[conteo < 5].index.tolist()
    if clases_eliminadas:
        print(f"[PIPELINE] Clases eliminadas por pocos ejemplos (< 5): {clases_eliminadas}")
        df = df[df["Emocion"].isin(clases_validas)].copy()
        print(f"[PIPELINE] Registros tras filtrar: {len(df)}")

    # Entrenar modelo, predecir y obtener confianzas
    predicciones, confianzas = entrenar_y_predecir(df, ruta_reporte=ruta_reporte)
    df["Prediccion_IA"] = predicciones
    df["Confianza_IA"] = confianzas
    print("[PIPELINE] Entrenamiento y prediccion finalizados.")

    # Comparativa
    coincidencias = (df["Emocion"] == df["Prediccion_IA"]).sum()
    total = len(df)
    conf_promedio = df["Confianza_IA"].mean()
    print(
        f"[PIPELINE] Coincidencias Etiqueta vs IA: "
        f"{coincidencias}/{total} ({coincidencias/total*100:.2f}%)"
    )
    print(f"[PIPELINE] Confianza promedio del modelo: {conf_promedio:.4f}")

    # Guardar en SQLite
    os.makedirs(ruta_db_dir, exist_ok=True)
    conn = sqlite3.connect(ruta_db)
    df.to_sql("tweets_procesados", conn, if_exists="replace", index=False)
    conn.close()
    print(f"[PIPELINE] Base de datos guardada en: {ruta_db}")
    print("[PIPELINE] Proceso completado.")


if __name__ == "__main__":
    main()
