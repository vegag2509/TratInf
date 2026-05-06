from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def entrenar_y_predecir(df, columna_texto="Tweet_Limpio", columna_etiqueta="Emocion", ruta_reporte=None):
    """
    Entrena un clasificador Logistic Regression usando TF-IDF sobre
    la columna de texto especificada. Devuelve las predicciones y la
    confianza (probabilidad) de cada prediccion sobre el DataFrame completo.

    Parametros:
        df (pd.DataFrame): DataFrame que contiene al menos las columnas
                           de texto y etiqueta.
        columna_texto (str): Nombre de la columna con el texto limpio.
        columna_etiqueta (str): Nombre de la columna con la etiqueta de emocion.
        ruta_reporte (str): Ruta opcional para guardar un archivo .txt con
                            las metricas del modelo.

    Retorna:
        tuple: (lista_predicciones, lista_confianzas)
    """
    X = df[columna_texto].astype(str)
    y = df[columna_etiqueta]

    # Validaciones basicas
    if X.empty or y.empty:
        raise ValueError("Las columnas de texto o etiqueta estan vacias.")

    clases_unicas = y.nunique()
    if clases_unicas < 2:
        print(f"[ADVERTENCIA] Solo se detecto {clases_unicas} clase(s). "
              "El modelo no puede entrenar con una sola clase; se devuelve "
              "la misma etiqueta para todas las filas.")
        iguales = [y.iloc[0]] * len(df)
        return iguales, [1.0] * len(df)

    # Split interno para evaluacion (80% entrenamiento / 20% prueba)
    conteo_clases = y.value_counts()
    clases_minimas = conteo_clases[conteo_clases < 2]
    uso_stratify = len(clases_minimas) == 0

    if not uso_stratify:
        print(f"[MODELO] Clases con muy pocos ejemplos detectadas: {list(clases_minimas.index)}. "
              "Usando split sin estratificacion.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    # Pipeline: vectorizacion TF-IDF + Logistic Regression
    modelo = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ])

    modelo.fit(X_train, y_train)

    # Evaluacion rapida
    y_pred_test = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    reporte = classification_report(y_test, y_pred_test, zero_division=0)

    print(f"[MODELO] Accuracy en conjunto de prueba: {acc:.4f}")
    print("[MODELO] Reporte de clasificacion:")
    print(reporte)

    # Guardar reporte en archivo si se solicita
    if ruta_reporte:
        lineas_reporte = [
            "=" * 60,
            "REPORTE DE ENTRENAMIENTO - DETECCION DE EMOCIONES",
            "=" * 60,
            f"Fecha y hora de ejecucion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Modelo: Logistic Regression + TF-IDF",
            f"Total de registros: {len(df)}",
            f"Registros entrenamiento: {len(X_train)}",
            f"Registros prueba: {len(X_test)}",
            f"Split estratificado: {'Si' if uso_stratify else 'No'}",
            "",
            "-" * 60,
            "DISTRIBUCION DE ETIQUETAS (DATASET COMPLETO)",
            "-" * 60,
        ]
        for emocion, cantidad in conteo_clases.items():
            lineas_reporte.append(f"  {emocion}: {cantidad}")

        lineas_reporte.extend([
            "",
            "-" * 60,
            "METRICAS EN CONJUNTO DE PRUEBA",
            "-" * 60,
            f"Accuracy: {acc:.4f}",
            "",
            reporte,
            "=" * 60,
            "FIN DEL REPORTE",
            "=" * 60,
        ])

        with open(ruta_reporte, "w", encoding="utf-8") as f:
            f.write("\n".join(lineas_reporte))
        print(f"[MODELO] Reporte guardado en: {ruta_reporte}")

    # Prediccion final sobre TODO el DataFrame (incluyendo train y test)
    predicciones = modelo.predict(X)

    # Obtener la confianza (probabilidad maxima) de cada prediccion
    probabilidades = modelo.predict_proba(X)
    confianzas = probabilidades.max(axis=1).round(4).tolist()

    return predicciones.tolist(), confianzas
