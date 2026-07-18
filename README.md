# Credit Risk Scoring v2 — Modelo de clasificación con threshold tuning basado en costes

Proyecto de scoring de riesgo crediticio construido sobre el German Credit Dataset. Es la segunda versión de un proyecto anterior (v1), reconstruida para corregir errores metodológicos identificados en una auditoría propia: data leakage en la elección del umbral, ausencia de métricas estándar del sector, y un bug crítico en la aplicación de despliegue.

## Problema de negocio

La concesión de crédito enfrenta a cualquier entidad financiera a un trade-off asimétrico: aprobar un préstamo a un cliente que no pagará genera una pérdida directa de capital; rechazar a un buen cliente genera un coste de oportunidad (el cliente se va a otro banco), pero no pérdida directa. Este proyecto construye un modelo que estima la probabilidad de impago de un solicitante a partir de seis variables, y aplica un umbral de decisión calibrado explícitamente sobre esa asimetría de costes, en lugar de usar el umbral por defecto de 0.5.

## Dataset

German Credit Dataset (OpenML, `credit-g`, versión 1). 1000 observaciones, 20 variables originales, recogidas en Alemania a mediados de los años 90.

**Limitaciones conocidas del dataset:**
- Tamaño reducido (1000 filas) para un problema con 6+ variables: muchas combinaciones específicas de variables están representadas por 0 o 1 casos reales.
- Antigüedad y origen geográfico: no representa el mercado crediticio español actual.
- Desbalanceo moderado del target: 70% buenos pagadores / 30% morosos.

## Selección de variables

De las 20 variables originales se seleccionaron 6, combinando tres criterios: sentido de negocio, disponibilidad realista en un formulario de solicitud, y ausencia de riesgo de discriminación:

- `duration` — duración del préstamo (meses)
- `credit_amount` — importe solicitado (€)
- `credit_history` — historial crediticio previo
- `checking_status` — estado de la cuenta corriente
- `savings_status` — nivel de ahorros
- `employment` — antigüedad en el empleo actual

**Variables descartadas explícitamente por riesgo de discriminación:**
- `personal_status` — combina estado civil y sexo; el sexo es una característica protegida.
- `foreign_worker` — proxy directo de nacionalidad/origen.
- `age` — característica protegida en zona gris legal; se optó por no incluirla para mantener el modelo libre de variables sensibles.

Tras aplicar One-Hot Encoding a las 4 variables categóricas (`drop_first=True` para evitar multicolinealidad), las 6 variables de negocio se expanden a **17 columnas** que recibe el modelo.

## Metodología

### Partición de datos

Split estratificado 80/20 (train/test) para mantener la proporción 70/30 del target en ambos conjuntos. Para el ajuste del umbral de decisión, el conjunto de train se subdivide adicionalmente en train (600) y validation (200), evitando así que la elección del umbral contamine la evaluación final en test.

### Escalado

Las dos variables numéricas (`duration`, `credit_amount`) se escalan con `StandardScaler`, ajustado (`fit_transform`) únicamente sobre train y aplicado (`transform`) sobre validation y test, para evitar fuga de información de conjuntos de evaluación hacia el proceso de entrenamiento.

### Modelos comparados

| Modelo | Recall (bad) | Precision (bad) | Accuracy |
|---|---|---|---|
| Logistic Regression (baseline) | 0.75 | 0.49 | 0.69 |
| Random Forest (challenger, umbral 0.5) | 0.52 | 0.60 | 0.75 |

El Random Forest obtiene mejor accuracy, pero peor recall sobre la clase de interés (morosos): con el umbral por defecto, deja sin detectar casi la mitad de los morosos reales. Esto ilustra por qué el accuracy es una métrica inadecuada en problemas desbalanceados, y motiva el ajuste del umbral de decisión sobre el Random Forest en lugar de descartarlo.

### Threshold tuning con matriz de costes

Se definió una matriz de costes explícita: coste de un falso negativo (moroso no detectado) = 7, coste de un falso positivo (buen cliente rechazado) = 3 — una ratio de aproximadamente 2.3:1, asumiendo que el impago cuesta más que el coste de oportunidad de un rechazo, pero sin descuidar el negocio comercial.

Se realizó un barrido de umbrales de 0.20 a 0.50 sobre el conjunto de validation, calculando el coste total esperado en cada punto:

| Umbral | Coste total (validation) |
|---|---|
| 0.20 | 263 (mínimo) |
| 0.25 | 266 |
| **0.30** | **267 (elegido)** |
| 0.35 | 271 |
| 0.40 | 292 |
| 0.50 | 338 |

El coste es prácticamente plano entre 0.20 y 0.35. Se eligió **0.30** en lugar del óptimo matemático (0.20) porque, a coste casi idéntico, reduce en un tercio los falsos positivos (40 frente a 62 buenos clientes rechazados), priorizando la relación con la clientela sin penalizar significativamente la detección de morosos.

### Resultado final (evaluado una única vez sobre test)

Con el umbral 0.30 aplicado sobre el conjunto de test (nunca utilizado en el proceso de ajuste):

- Recall (bad): **0.78** — se detecta el 78% de los morosos reales.
- Precision (bad): 0.54
- Accuracy: 0.73
- Matriz de confusión: 47 morosos detectados, 13 no detectados, 40 buenos clientes rechazados, 100 buenos clientes aprobados.

El resultado en test iguala o mejora al de validation, lo que indica que el modelo generaliza razonablemente y que el umbral no está sobreajustado al conjunto usado para calibrarlo.

## Métricas de discriminación

- **ROC-AUC: 0.814** — buena capacidad de ordenar correctamente el riesgo relativo entre clientes, en línea con lo esperable para un modelo con 6 variables.
- **KS statistic: 0.519** — considerado excelente en el contexto de scoring bancario (referencia habitual del sector: >0.5 es un resultado sólido).

Ambas métricas son coherentes entre sí y con el resto de resultados, lo que da consistencia al análisis.

## Interpretabilidad del modelo

Se calculó la importancia de variables con dos métodos, dado que el método por defecto de Random Forest (Mean Decrease in Impurity) tiene un sesgo conocido hacia variables numéricas con muchos valores únicos:

- **MDI (impureza):** `credit_amount` domina (34%), seguido de `duration` (22%).
- **Permutation importance** (más robusta frente a ese sesgo): `duration` pasa a ser la variable más influyente, seguida de `checking_status_no checking`; la importancia de `credit_amount` cae drásticamente (de 34% a ~3%), revelando que estaba sobreestimada por el método MDI.

Esta discrepancia se documenta explícitamente porque afecta a qué variables se presentarían como "más relevantes" ante un regulador o un comité de riesgos.

## Aplicación (Streamlit)

`app.py` implementa un simulador interactivo con las 6 variables del modelo. Corrige un bug crítico presente en la versión anterior del proyecto: la v1 rellenaba con ceros las variables categóricas no introducidas por el usuario, lo que producía predicciones sobre un perfil de cliente inexistente en los datos de entrenamiento (sin cuenta, sin historial, sin ahorros, sin empleo, sin propósito). La v2 construye explícitamente la fila de datos del cliente, aplica el mismo encoding utilizado en entrenamiento, y usa `reindex` contra la lista de columnas del modelo para garantizar una estructura idéntica a la esperada — sin variables inventadas ni valores implícitos incorrectos.

Los rangos de `duration` (≤48 meses) y `credit_amount` (≤10.000€) se limitaron explícitamente en el formulario, basándose en la representación real de esos rangos en el dataset de entrenamiento (por encima de esos valores, la representación cae por debajo del 2-4% de los casos).

## Limitaciones

- **Tamaño del dataset:** 1000 observaciones son insuficientes para cubrir con garantías todas las combinaciones posibles de 6 variables. Se comprobó, por ejemplo, que un perfil concreto con cuatro señales de riesgo simultáneas (sin cuenta corriente, sin ahorros, historial crediticio crítico, desempleado) solo está representado por **un único caso** en todo el dataset. Las predicciones sobre combinaciones de variables poco representadas deben interpretarse con cautela, independientemente de que cada variable individual esté dentro de rangos razonables.
- **Antigüedad y origen del dataset:** no reflejan el mercado crediticio español actual; el proyecto tiene valor metodológico y de portfolio, no de despliegue en producción real.
- **Ausencia de variables de comportamiento transaccional:** el modelo no incorpora datos de comportamiento reciente (movimientos, patrones de gasto), que en un entorno real complementarían significativamente el scoring.
- **class_weight='balanced' no es neutral:** desplaza la frontera de decisión hacia una mayor detección de morosos a costa de más falsos positivos; la elección del umbral posterior corrige parcialmente este efecto, pero la interacción entre ambos mecanismos debe entenderse como conjunta, no independiente.

## Próximos pasos (fuera del alcance actual)

- Calcular el coste de la matriz sobre datos reales de la entidad, en lugar de un supuesto (7:3).
- Explorar variables ordinales para `credit_history` en lugar de One-Hot Encoding, para reducir la fragmentación de importancia entre sus categorías.
- Monitorización de drift si el modelo se desplegara: seguimiento de PSI (Population Stability Index) entre la población de entrenamiento y la población real de solicitantes.

## Stack técnico

Python 3, pandas, numpy, scikit-learn (LogisticRegression, RandomForestClassifier, StandardScaler, métricas), scipy (KS statistic), Streamlit (aplicación de despliegue), joblib (serialización de modelo).

## Cómo ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

El modelo, el scaler, la lista de columnas y el umbral se cargan desde los archivos `.pkl` generados al final del notebook `credit_risk_v2.ipynb`. Para regenerarlos, ejecutar el notebook completo de principio a fin.
