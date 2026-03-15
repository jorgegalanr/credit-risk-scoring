# 🏦 Credit Risk Scoring: Modelo de Clasificación Bancaria

Este es el tercer y último proyecto de mi portfolio **Mastering Financial Data Science**, diseñado para cubrir la "Santísima Trinidad" de las finanzas: Mercados, Tesorería y, en este caso, **Riesgo Crediticio**.

En este proyecto, actúo como Analista de Riesgos (*Risk Data Scientist*), desarrollando un algoritmo de Machine Learning capaz de evaluar solicitudes de crédito y predecir la probabilidad de impago (*default*) de los clientes.

## 🎯 El Caso de Negocio

La concesión de créditos es el motor (y el mayor riesgo) de cualquier entidad financiera. Este proyecto resuelve un problema crítico: **¿Cómo automatizar la aprobación de préstamos minimizando la morosidad sin paralizar el negocio comercial?**

Utilizando el estándar de la industria (el *German Credit Dataset*, adaptado al contexto financiero en español), el modelo analiza variables demográficas, financieras y de comportamiento para tomar decisiones de aprobación o denegación en milisegundos.

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python 3
* **Manipulación de Datos:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (Random Forest Classifier, train_test_split, metrics)
* **Visualización:** `matplotlib`, `seaborn`

## 📊 Fases del Proyecto y Resultados Clave

### 1. Análisis Exploratorio y Desbalanceo
La auditoría inicial de los datos reveló una cartera altamente desbalanceada con un **30% de morosidad** base. Se detectó que el volumen principal de préstamos se destina a bienes de consumo (tecnología y vehículos). Se aplicó *One-Hot Encoding* para transformar las variables categóricas en una matriz matemática de 39 dimensiones.

### 2. Modelo Base (Random Forest)
Se entrenó un modelo *Random Forest* inicial. Si bien su precisión general era buena, el análisis de la **Matriz de Confusión** demostró que era demasiado permisivo desde el punto de vista del riesgo bancario, permitiendo un alto número de Falsos Negativos (morosos a los que se les concedía el préstamo por error).

### 3. Ajuste de Políticas de Riesgo (Threshold Tuning)
En banca, el coste de un Falso Negativo (dinero perdido) es infinitamente superior al de un Falso Positivo (coste de oportunidad). 
* Se intervino el umbral de decisión matemática del algoritmo.
* **Política Estricta:** Se obligó a la IA a denegar automáticamente cualquier préstamo que presentara más de un **30% de riesgo de impago**.
* **Impacto Financiero:** Esta decisión **redujo la entrada de morosos en más de un 50%** (pasando de 36 a solo 15 impagos colados en el set de prueba), protegiendo el capital de la entidad frente a escenarios macroeconómicos adversos.

### 4. Explicabilidad del Modelo (Feature Importance)
Para evitar el efecto "caja negra" y cumplir con la transparencia exigida por los reguladores, se extrajo el "cerebro" del algoritmo. Las 3 variables que más ponderan en la decisión de crédito son:
1. **Importe Solicitado:** La exposición al riesgo total.
2. **Edad:** Correlacionado con la madurez y estabilidad laboral.
3. **Duración en Meses:** El plazo temporal en el que pueden ocurrir imprevistos económicos.


