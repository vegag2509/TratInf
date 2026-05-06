import re
import emoji


def limpiar_texto(texto):
    """
    Limpia un string aplicando las siguientes reglas:
      1. Reemplaza menciones de usuarios (@usuario) por la etiqueta '[USER]'.
      2. Elimina completamente cualquier URL.
      3. Convierte emojis a su representacion textual (:emoji_name:).
      4. Elimina espacios en blanco redundantes y strip final.

    Parametros:
        texto (str): Texto original a limpiar.

    Retorna:
        str: Texto limpio y normalizado.
    """
    # Manejar nulos
    if not isinstance(texto, str):
        texto = str(texto) if texto is not None else ""

    # 1) Reemplazar menciones de usuario
    texto = re.sub(r"@\w+", "[USER]", texto)

    # 2) Eliminar URLs (http, https, www)
    texto = re.sub(r"https?://\S+|www\.\S+", "", texto)

    # 3) Convertir emojis a texto descriptivo
    # Se intenta en espanol; si falla se usa el default (ingles)
    try:
        texto = emoji.demojize(texto, language="es")
    except (TypeError, ValueError):
        texto = emoji.demojize(texto)

    # 4) Normalizar espacios en blanco
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto
