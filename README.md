# Proyecto Final de MLOps

## Automatización de un pipeline de Machine Learning con GitHub Actions y MLflow

Este proyecto implementa un pipeline reproducible de Machine Learning que carga un dataset externo, realiza preprocesamiento, entrena un modelo, evalúa métricas y registra el experimento con MLflow. La ejecución principal del proyecto está orientada a la nube mediante **GitHub Actions**, por lo que el flujo no depende de la configuración local del computador del usuario.

## 1. Dataset utilizado

Se utiliza el dataset externo **Wine Quality - Red Wine** del repositorio UCI Machine Learning Repository.

Fuente del dataset:

```text
https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
```

Este dataset contiene variables fisicoquímicas de vinos tintos, como acidez, azúcar residual, cloruros, dióxido de azufre, densidad, pH, sulfatos y alcohol. La variable original `quality` se transforma en una variable binaria:

- `1`: vino con calidad aceptable o alta, si `quality >= 6`.
- `0`: vino con calidad baja, si `quality < 6`.

No se utiliza `sklearn.datasets`, cumpliendo el requisito de trabajar con una fuente externa.

## 2. Estructura del proyecto

```text
.
├── .github/
│   └── workflows/
│       └── ml.yml
├── artifacts/
│   └── .gitkeep
├── data/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   └── train.py
├── tests/
│   └── test_config.py
├── config.yaml
├── Makefile
├── README.md
└── requirements.txt
```

## 3. Preprocesamiento

El pipeline realiza los siguientes pasos:

1. Carga del dataset desde una URL externa.
2. Conversión de la variable `quality` en una variable objetivo binaria.
3. Separación de datos en entrenamiento y prueba.
4. Imputación de valores faltantes mediante la mediana.
5. Escalamiento de variables numéricas con `StandardScaler`.
6. Entrenamiento del modelo dentro de un `Pipeline` de scikit-learn.

## 4. Modelo utilizado

Se utiliza un modelo `RandomForestClassifier` porque es adecuado para problemas de clasificación tabular, permite capturar relaciones no lineales y suele tener buen desempeño sin requerir demasiada configuración inicial.

Los hiperparámetros principales están definidos en `config.yaml`:

```yaml
n_estimators: 150
max_depth: 8
random_state: 42
class_weight: balanced
```

## 5. Métricas de evaluación

El proyecto calcula y registra las siguientes métricas:

- Accuracy
- Precision
- Recall
- F1-score

Estas métricas se guardan en MLflow y también en el archivo:

```text
artifacts/metrics.json
```

## 6. Registro de experimentos con MLflow

El proyecto utiliza **MLflow Tracking** para registrar el experimento de Machine Learning durante la ejecución del pipeline.

Durante el entrenamiento se registran:

- Parámetros del dataset.
- Hiperparámetros del modelo.
- Métricas de evaluación.
- Firma del modelo con `infer_signature`.
- Ejemplo de entrada con `input_example`.
- Modelo entrenado como artefacto mediante `mlflow.sklearn.log_model`.

El tracking se guarda en la carpeta:

```text
mlruns/
```

En este proyecto, la carpeta `mlruns/` se genera automáticamente durante la ejecución del workflow en **GitHub Actions** y se conserva como parte de los artefactos del workflow.

## 7. Automatización con GitHub Actions

El pipeline de CI/CD se encuentra definido en:

```text
.github/workflows/ml.yml
```

Este workflow se ejecuta automáticamente cuando se realiza un `push` o un `pull_request` sobre las ramas `main` o `master`. También puede ejecutarse manualmente desde la pestaña **Actions** mediante `workflow_dispatch`.

El workflow ejecuta las siguientes tareas:

```bash
make install
make lint
make test
make train
```

Al finalizar correctamente, GitHub Actions guarda como artefacto del workflow:

```text
mlops-model-artifacts
```

Este artefacto contiene los resultados generados por el pipeline, incluyendo:

- Carpeta `artifacts/`.
- Carpeta `mlruns/`.
- Archivo `artifacts/metrics.json`.
- Modelo entrenado `artifacts/model/model.joblib`.
- Evidencia del tracking con MLflow.

## 8. Ejecución del proyecto en la nube

La ejecución principal del proyecto se realiza en la nube mediante **GitHub Actions**.

Cada vez que se realiza un `push` al repositorio o se ejecuta manualmente el workflow desde la pestaña **Actions**, GitHub Actions realiza automáticamente las siguientes etapas:

1. Clona el repositorio.
2. Configura Python 3.10.
3. Instala las dependencias del proyecto.
4. Ejecuta validación de estilo del código con `ruff`.
5. Ejecuta pruebas básicas con `pytest`.
6. Entrena el modelo de Machine Learning.
7. Registra parámetros, métricas, firma, `input_example` y modelo con MLflow.
8. Guarda los artefactos generados por el pipeline.
