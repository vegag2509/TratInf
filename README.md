# Explicacion Completa del Algoritmo: Logistic Regression + TF-IDF

## Resumen

Este proyecto detecta emociones en tweets usando un algoritmo de Machine Learning. Funciona en 4 pasos:

1. **Limpieza:** Quita basura del texto (URLs, usuarios, convierte emojis a palabras).
2. **TF-IDF:** Transforma cada palabra en un numero segun que tan importante y que tan rara es.
3. **Logistic Regression:** Un modelo matematico que aprende patrones. Cada palabra tiene un "peso" para cada emocion. Al principio los pesos son aleatorios, pero el modelo los corrige poco a poco viendo miles de ejemplos, hasta que sabe reconocer emociones.
4. **Prediccion:** Para cada tweet nuevo, el modelo calcula un puntaje por cada emocion, lo convierte en probabilidad (0 a 1) y devuelve la emocion mas probable junto con su nivel de confianza.

**Ajuste clave:** El modelo ignora automaticamente emociones con menos de 5 tweets (como Asco y Miedo) porque no se pueden aprender con tan pocos ejemplos.

---

## Parte 1: El problema

Tenemos miles de tweets de Jordi Wild. Algunos estan felices, otros enojados, otros tristes. Queremos que una computadora lea cada tweet y nos diga que emocion expresa.

El problema: las computadoras no entienden palabras. Solo entienden numeros.

Por eso el algoritmo hace dos cosas:
1. Convierte palabras a numeros (TF-IDF).
2. Usa esos numeros para calcular la probabilidad de cada emocion (Logistic Regression).

---

## Parte 2: TF-IDF (De palabras a numeros)

### 2.1. TF = Frecuencia de la palabra en este tweet

Cuenta cuantas veces aparece cada palabra en un solo tweet.

Ejemplo:
- Tweet A: "odio odio esto"
  - "odio" aparece 2 veces
  - "esto" aparece 1 vez

- Tweet B: "me encanta la vida"
  - "me" aparece 1 vez
  - "encanta" aparece 1 vez
  - "vida" aparece 1 vez

### 2.2. IDF = Que tan rara es la palabra en TODOS los tweets

Palabras comunes como "el", "la", "que" aparecen en casi todos los tweets. No sirven para distinguir emociones.

Palabras raras como "odio", "miedo", "llorar" aparecen en muy pocos tweets. Son muy valiosas.

IDF les da un valor alto a las palabras raras, y un valor bajo a las palabras comunes.

Ejemplo en nuestro dataset:

| Palabra | Aparece en cuantos tweets | Valor IDF |
|---|---|---|
| "el" | 700 de 759 | Bajo (0.1) |
| "dia" | 300 de 759 | Medio (0.5) |
| "odio" | 15 de 759 | Alto (2.8) |
| "miedo" | 3 de 759 | Muy alto (4.1) |

### 2.3. TF-IDF = TF multiplicado por IDF

TF-IDF de una palabra = (cuantas veces aparece aqui) x (que tan rara es en todos lados)

Ejemplo concreto:

Palabra "odio" en un tweet de enojo:
- TF = 2 (aparece 2 veces)
- IDF = 2.8 (es rara)
- TF-IDF = 2 x 2.8 = 5.6

Palabra "el" en ese mismo tweet:
- TF = 1
- IDF = 0.1 (es comun)
- TF-IDF = 1 x 0.1 = 0.1

Resultado: "odio" vale 56 veces mas que "el". La computadora ya sabe que palabras importan.

### 2.4. El vector final

Cada tweet se convierte en una lista de miles de numeros, uno por cada palabra distinta del dataset.

Es como una "huella digital numerica". Dos tweets de la misma emocion tendran huellas parecidas. Dos tweets de emociones distintas tendran huellas diferentes.

---

## Parte 3: Logistic Regression (La calculadora de emociones)

Ya tenemos cada tweet como una lista de numeros. Ahora necesitamos una formula que decida la emocion.

### 3.1. Los pesos: la tabla secreta del modelo

El modelo crea una tabla donde cada palabra tiene un "peso" para cada emocion.

Al principio, todos los pesos son numeros aleatorios. El modelo no sabe nada.

Tabla inicial (aleatoria):

| Palabra | Peso para Enfado | Peso para Alegría | Peso para Neutral |
|---|---|---|---|
| "odio" | 0.12 | -0.05 | 0.33 |
| "encanta" | -0.21 | 0.08 | -0.14 |
| "maldito" | 0.05 | 0.19 | -0.02 |
| "feliz" | -0.10 | 0.15 | 0.05 |

### 3.2. Como se calcula el peso: el entrenamiento paso a paso

El modelo aprende viendo ejemplos y corrigiendo errores.

#### Paso A: Mostrar un tweet de entrenamiento

Tweet: "odio esto, me encanta la vida"
Emocion real (correcta): Alegría

#### Paso B: El modelo adivina con los pesos actuales

Calcula un "score" para cada emocion multiplicando los TF-IDF de las palabras por sus pesos.

Para Enfado:
= (TF-IDF de "odio" x peso de "odio" para Enfado) + (TF-IDF de "encanta" x peso de "encanta" para Enfado) + ...
= (5.6 x 0.12) + (4.1 x -0.21) + ...
= +0.67 - 0.86 + ...
= +1.2

Para Alegría:
= (5.6 x -0.05) + (4.1 x 0.08) + ...
= -0.28 + 0.33 + ...
= +0.8

Para Neutral:
= (5.6 x 0.33) + (4.1 x -0.14) + ...
= +1.85 - 0.57 + ...
= -0.3

Resultado de la prediccion:

| Emoción | Score | Probabilidad (Softmax) |
|---|---|---|
| Enfado | +1.2 | 45% ← predice esto |
| Alegría | +0.8 | 35% |
| Neutral | -0.3 | 20% |

El modelo dijo Enfado, pero la respuesta correcta era Alegría. Se equivoco.

#### Paso C: Calcular el error

El modelo compara lo que dijo contra lo que debio decir:

- Alegría debio ser 100%, pero fue 35%. Error grande.
- Enfado debio ser 0%, pero fue 45%. Error grande.

#### Paso D: Ajustar los pesos (la correccion)

El modelo mira que palabras causaron el error y les cambia el peso.

**Palabra "odio":**
- Aparecio en el tweet.
- El tweet era Alegría, pero "odio" le dio puntos a Enfado.
- Eso fue malo.
- El algoritmo baja el peso de "odio" para Enfado: de 0.12 pasa a 0.08.
- El algoritmo sube el peso de "odio" para Alegría: de -0.05 pasa a +0.01.

**Palabra "encanta":**
- Aparecio en el tweet.
- El tweet era Alegría, pero "encanta" le dio pocos puntos a Alegría.
- El algoritmo sube el peso de "encanta" para Alegría: de 0.08 pasa a 0.25.
- El algoritmo baja el peso de "encanta" para Enfado: de -0.21 pasa a -0.30.

**Palabra "maldito":**
- No estaba en este tweet, asi que no se toca.

Tabla de pesos despues de corregir este tweet:

| Palabra | Peso para Enfado | Peso para Alegría | Peso para Neutral |
|---|---|---|---|
| "odio" | ~~0.12~~ → 0.08 | ~~-0.05~~ → +0.01 | 0.33 |
| "encanta" | ~~-0.21~~ → -0.30 | ~~+0.08~~ → +0.25 | -0.14 |

Los numeros se movieron un poquito en la direccion correcta.

#### Paso E: Repetir miles de veces

El algoritmo hace esto con los 607 tweets de entrenamiento, una y otra vez:

1. Muestra un tweet.
2. Predice con los pesos actuales.
3. Compara contra la respuesta real.
4. Calcula el error.
5. Ajusta los pesos un poquito.
6. Pasa al siguiente tweet.

Despues de ver cientos de tweets de entrenamiento, los pesos ya no son aleatorios. Han sido corregidos miles de veces.

#### Paso F: La formula real (Gradient Descent)

En la practica, el ajuste no lo hace un humano. Lo hace una formula matematica llamada "Descenso del Gradiente".

Es como estar en una montaña a oscuras con una literna que solo ilumina tus pies:
- Calculas en que direccion esta la bajada mas empinada (el gradiente).
- Das un paso pequeno en esa direccion.
- Repites.

El "valle" de la montaña es el punto donde los errores son minimos. El algoritmo camina poco a poco hacia ese valle.

El tamano del paso se llama "learning rate". Scikit-learn lo calcula automaticamente.

### 3.3. Pesos entrenados (despues del aprendizaje)

Al final del entrenamiento, los pesos se ven asi:

| Palabra | Peso para Enfado | Peso para Alegría | Peso para Neutral |
|---|---|---|---|
| "odio" | **+3.5** | **-2.1** | **-1.8** |
| "encanta" | **-2.2** | **+3.9** | **-1.1** |
| "maldito" | **+2.8** | **-1.5** | **-0.9** |
| "feliz" | **-1.9** | **+3.2** | **-0.5** |
| "llorar" | **+0.5** | **-1.8** | **-0.3** |
| "el" | **+0.1** | **+0.1** | **+0.2** |

Que significan estos numeros?

- "odio" tiene peso +3.5 para Enfado. Cada vez que aparece, empuja fuerte hacia Enfado.
- "odio" tiene peso -2.1 para Alegría. Cada vez que aparece, le quita puntos a Alegría.
- "el" tiene pesos casi iguales y bajos para todos. No importa para ninguna emocion.

### 3.4. De scores a probabilidades: la funcion Softmax

El score es solo un numero. Pero queremos una probabilidad: que tan seguro estamos?

Para eso se usa Softmax. Toma los 5 scores (uno por emocion) y los convierte en 5 probabilidades que suman 100%.

Regla intuitiva:
- Si el score es positivo y grande, la probabilidad es cercana a 1 (100%).
- Si el score es negativo y grande, la probabilidad es cercana a 0 (0%).
- Si el score es cercano a 0, la probabilidad es de 0.5 (50%, totalmente inseguro).

Ejemplo:

| Emoción | Score | Probabilidad |
|---|---|---|
| Enfado | +31.4 | 0.9999 (99.99%) |
| Alegría | -18.1 | 0.0000 (0.00%) |
| Neutral | -13.9 | 0.0001 (0.01%) |
| Sorpresa | -5.2 | 0.0000 (0.00%) |
| Tristeza | -20.5 | 0.0000 (0.00%) |

En este caso el modelo esta completamente seguro de que es Enfado.

Otro ejemplo mas realista:

| Emoción | Score | Probabilidad |
|---|---|---|
| Enfado | +2.5 | 0.62 |
| Tristeza | +1.1 | 0.25 |
| Neutral | +0.3 | 0.08 |
| Sorpresa | -0.8 | 0.03 |
| Alegría | -1.5 | 0.02 |

Prediccion: Enfado (62% de confianza).

Esa columna "Confianza_IA" del proyecto es exactamente esa probabilidad maxima (0.62 en este ejemplo).

---

## Parte 4: Evaluacion (El examen sorpresa)

El modelo entreno con el 80% de los tweets (607). Ahora hay que evaluarlo con el 20% restante (152) que NUNCA vio.

Por que? Porque si le preguntas lo que ya estudio, va a acertar todo. Eso no significa que aprendio.

Ejemplo:

| Tweet (de prueba) | Emocion real | Prediccion del modelo | ¿Acierto? |
|---|---|---|---|
| "me encanta la vida" | Alegría | Alegría | SI |
| "odio esto" | Enfado | Neutral | NO |
| "estoy triste hoy" | Tristeza | Tristeza | SI |
| "no me lo puedo creer" | Sorpresa | Alegría | NO |

Si acierta 96 de 152:
Accuracy = 96 / 152 = 0.6316 = 63.16%

Esa es la metrica honesta.

---

## Parte 5: Ejemplo completo de principio a fin

### Tweet original:
"Maldito Miyazaki hijo de puta lo has vuelto a hacer…"

### Paso 1: Limpieza
"Maldito Miyazaki hijo de [USER] lo has vuelto a hacer"

### Paso 2: TF-IDF
- "maldito" -> TF-IDF = 3.2 (palabra rara, muy emotiva)
- "hijo" -> TF-IDF = 0.4
- "vuelto" -> TF-IDF = 0.3
- "hacer" -> TF-IDF = 0.1

### Paso 3: Logistic Regression calcula scores

Para Enfado:
= (3.2 x 3.5) + (0.4 x 0.2) + (0.3 x 0.1) + (0.1 x -0.1)
= 11.2 + 0.08 + 0.03 - 0.01
= +11.3

Para Alegría:
= (3.2 x -2.1) + (0.4 x 0.1) + (0.3 x 0.0) + (0.1 x 0.1)
= -6.72 + 0.04 + 0 + 0.01
= -6.67

Para Neutral:
= (3.2 x -0.8) + (0.4 x 0.3) + ...
= -2.56 + 0.12 + ...
= -2.1

Para Sorpresa:
= (3.2 x 0.1) + ...
= +0.5

Para Tristeza:
= (3.2 x -0.3) + ...
= -0.9

### Paso 4: Softmax a probabilidades

| Emoción | Probabilidad |
|---|---|
| Enfado | 0.94 |
| Sorpresa | 0.03 |
| Neutral | 0.02 |
| Alegría | 0.01 |
| Tristeza | 0.00 |

### Resultado final:
- Prediccion_IA = Enfado
- Confianza_IA = 0.94

El modelo esta 94% seguro de que este tweet expresa enojo. Y tiene razon.

---

## Parte 6: Ajuste clave - class_weight='balanced'

Nuestro dataset esta desbalanceado:

| Emoción | Tweets |
|---|---|
| Neutral | 400 |
| Alegría | 253 |
| Enfado | 51 |
| Sorpresa | 29 |
| Tristeza | 24 |

Si el modelo es perezoso, puede decir Neutral para todo y ya acierta 400 de 757 (52%). Eso es una trampa.

class_weight='balanced' le dice al algoritmo:
"Aciertar un tweet de Enfado vale 8 veces mas que acertar uno de Neutral. Aciertar uno de Tristeza vale 16 veces mas."

Esto fuerza al modelo a prestar atencion a las clases pequenas, aunque le cueste acierto en las grandes.

---

## Resumen visual

```
TWEET ORIGINAL
      |
      v
[Limpieza] ---> quita URLs, @usuarios, emojis->texto
      |
      v
[TF-IDF] -----> convierte cada palabra en un numero
      |
      v
[Logistic Regression]
   |
   |-- Calcula 5 scores (uno por emocion)
   |   usando los pesos aprendidos en el entrenamiento
   |
   |-- Aplica Softmax -> 5 probabilidades
   |
   |-- Elige la probabilidad mayor
   |
      v
PREDICCION: "Enfado"
CONFIANZA:  0.94
```
