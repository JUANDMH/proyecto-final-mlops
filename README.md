# Proyecto Final de MLOps

## Automatización de un pipeline de Machine Learning con GitHub Actions y MLflow

Este proyecto implementa un pipeline reproducible de Machine Learning que carga un dataset externo, realiza preprocesamiento, entrena un modelo, evalúa métricas y registra el experimento con MLflow. Además, automatiza la ejecución del flujo mediante GitHub Actions.

## 1. Dataset utilizado

Se utiliza el dataset externo **Wine Quality - Red Wine** del repositorio UCI Machine Learning Repository.

Fuente del dataset:

`https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv`

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
├── data/
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

Estas métricas se guardan en MLflow y también en el archivo local:

```text
artifacts/metrics.json
```

## 6. Uso de MLflow

El script `src/train.py` registra en MLflow:

- Parámetros del dataset.
- Hiperparámetros del modelo.
- Métricas de evaluación.
- Firma del modelo con `infer_signature`.
- Ejemplo de entrada con `input_example`.
- Modelo entrenado con `mlflow.sklearn.log_model`.

El tracking se realiza localmente en:

```text
mlruns/
```

Para visualizar los experimentos:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Luego abrir en el navegador:

```text
http://127.0.0.1:5000
```

## 7. Ejecución local

### Instalar dependencias

```bash
make install
```

### Ejecutar pruebas

```bash
make test
```

### Entrenar el modelo

```bash
make train
```

También se puede ejecutar directamente:

```bash
python src/train.py
```

## 8. Automatización con GitHub Actions

El archivo `.github/workflows/ml.yml` ejecuta automáticamente el pipeline cuando se hace un `push`, un `pull_request` o cuando se lanza manualmente desde GitHub Actions.

El workflow realiza estos pasos:

1. Clona el repositorio.
2. Configura Python.
3. Instala dependencias con `make install`.
4. Valida estilo de código con `make lint`.
5. Ejecuta pruebas con `make test`.
6. Entrena el modelo con `make train`.
7. Guarda como artefactos:
   - Carpeta `artifacts/`
   - Carpeta `mlruns/`
