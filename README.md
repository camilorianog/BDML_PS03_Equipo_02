# Problem Set 03: Making Money with ML?

## Grupo 2 | Big Data and Machine Learning para Economía Aplicada

MECA 4107 — Universidad de los Andes — 2026-10

---

# Autores

| Nombre            | Código    |
| ----------------- | --------- |
| Jose A. Rincon S  | 202013328 |
| Juan C. Riaño     | 202013305 |
| Lucas Rodriguez   | 202021985 |
| Santiago Gonzalez | 202110234 |

---

# Descripción

Este repositorio contiene el desarrollo del **Problem Set 03** del curso *Big Data and Machine Learning para Economía Aplicada*. El objetivo es construir un modelo predictivo para estimar precios de vivienda en Bogotá D.C., con énfasis en propiedades ubicadas en Chapinero, utilizando información estructural de los inmuebles, variables espaciales derivadas de OpenStreetMap (OSM) y señales obtenidas desde texto libre de los anuncios inmobiliarios.

El problema está inspirado en el caso de Zillow Offers, donde errores sistemáticos de sobrepredicción generaron pérdidas cercanas a USD 500 millones. En este contexto, el desafío no consiste únicamente en minimizar error predictivo, sino en desarrollar un modelo suficientemente robusto para apoyar decisiones reales de compra de vivienda bajo costos asimétricos:

* **Overprediction** → destruye capital (comprar caro)
* **Underprediction** → implica perder una oportunidad de negocio

La pregunta central del proyecto es:

> **¿Cuál es el mejor modelo para predecir precios de vivienda en Chapinero y se puede confiar lo suficiente en él para comprar?**

---

# Objetivo del Proyecto

Construir y evaluar modelos de predicción de precios de vivienda capaces de:

* Generalizar desde otras localidades de Bogotá hacia Chapinero
* Incorporar señales espaciales y urbanas externas
* Aprovechar información contenida en texto libre
* Minimizar errores económicamente costosos
* Evaluar robustez mediante validación espacial

---

# Instrucciones de Replicación

## Requisitos

* R ≥ 4.3.0
* RStudio recomendado
* Kaggle API configurado (opcional)

---

## 1. Descargar datos de la competencia

```r
# kaggle competitions download -c uniandes-bdml-2026-10-ps3
```

Los archivos de la competencia deben almacenarse en:

```text
00_data/00_raw/00_competition/
```

Archivos esperados:

```text
train.csv
test.csv
submission_template.csv
```

---

## 2. Descargar datos geoespaciales externos

Además de los datos originales de Properati, el proyecto incorpora múltiples capas geoespaciales oficiales de Bogotá D.C. utilizadas para construir variables espaciales, administrativas y de validación territorial.

Los shapefiles deben descargarse manualmente y almacenarse en:

```text
00_data/00_raw/
```

---

### 2.1 Estratificación Socioeconómica

Fuente oficial:

https://datosabiertos.bogota.gov.co/dataset/estratificacion-para-bogota

Archivo utilizado:

```text
ManzanaEstratificacion.shp
```

Uso dentro del pipeline:

* asignación de estrato mediante spatial join
* construcción de variables socioeconómicas espaciales
* análisis de sesgo por estrato

---

### 2.2 UPZ — Unidades de Planeamiento Zonal

Fuente oficial:

https://datosabiertos.bogota.gov.co/en/dataset/espacio-publico-total-upz-2021

Archivo utilizado:

```text
EPT_UPZ.shp
```

Uso dentro del pipeline:

* validación espacial
* agrupación territorial
* spatial cross-validation
* análisis de error por zonas

---

### 2.3 Localidades de Bogotá D.C.

Fuente oficial:

https://datosabiertos.bogota.gov.co/dataset/localidad-bogota-d-c

Archivo utilizado:

```text
Loca.shp
```

Uso dentro del pipeline:

* joins administrativos
* análisis territorial del error
* construcción de variables de localización

---

## 3. Ejecutar pipeline completo

```r
source("00_rundirectory.R")
```

El script maestro ejecuta secuencialmente:

1. Limpieza de datos
2. Construcción de variables textuales
3. Construcción de variables espaciales
4. Imputación y creación de flags
5. Construcción de folds espaciales
6. Entrenamiento de modelos
7. Validación cruzada
8. Generación de submissions
9. Producción de tablas y figuras finales

---

# Estructura del Repositorio

```text
.
├── 00_rundirectory.R
├── BDML_PS03_Equipo_02.Rproj
├── README.md
│
├── 00_data/
│
│   ├── 00_raw/
│   │   ├── 00_competition/
│   │   │   ├── train.csv
│   │   │   ├── test.csv
│   │   │   └── submission_template.csv
│   │   │
│   │   ├── EPT_UPZ.*
│   │   ├── Loca.*
│   │   ├── ManzanaEstratificacion.*
│   │   │
│   │   └── osm_cache/
│   │       ├── amenity_bank.rds
│   │       ├── amenity_bus_station.rds
│   │       ├── amenity_cafe.rds
│   │       ├── amenity_hospital.rds
│   │       ├── amenity_pharmacy.rds
│   │       ├── amenity_restaurant.rds
│   │       ├── amenity_school.rds
│   │       ├── amenity_university.rds
│   │       ├── highway_street_lamp.rds
│   │       ├── landuse_residential_poly.rds
│   │       ├── leisure_fitness_centre.rds
│   │       ├── leisure_park_poly.rds
│   │       ├── railway_station.rds
│   │       └── shop_supermarket.rds
│   │
│   ├── 01_process/
│   │   ├── 00_clean.R
│   │   ├── 01_text_variables.R
│   │   ├── 02_spatial_variables.R
│   │   ├── 02a_spatial_variables_assets/
│   │   │   ├── 01_estratos/    (ManzanaEstratificacion.*)
│   │   │   ├── 02_UPZ/         (EPT_UPZ.*)
│   │   │   └── 03_localidades/ (Loca.*)
│   │   ├── 03_imputation.R
│   │   └── 04_cv_setup.R
│   │
│   ├── 01_processed/
│   │   ├── train_model.rds
│   │   └── test_model.rds
│   │
│   └── 02_cv/
│       ├── folds_global.rds
│       ├── folds_norte.rds
│       ├── folds_spatial.rds
│       ├── folds_spatial_upz.rds
│       ├── folds_std.rds
│       ├── pesos_dist_6km.rds
│       └── pesos_uniform.rds
│
├── 01_R/
│   └── 00_functions/
│       ├── 00_nm.R
│       ├── 01_log_modelo.R
│       └── 02_generar_submission.R
│
├── 02_models/
│
│   ├── 00_training/
│   │   ├── 00_linear_regression.R
│   │   ├── 01_elastic_net.R
│   │   ├── 02_regression_trees.R
│   │   ├── 03_random_forest.R
│   │   ├── 04_boosting.R
│   │   ├── 05_neural_network.R
│   │   └── 06_super_learning.R
│   │
│   └── 01_submissions/
│       ├── 00_linear_regression/
│       ├── 01_elastic_net/
│       ├── 02_regression_trees/
│       ├── 03_random_forest/
│       ├── 04_boosting/
│       ├── 05_neural_networks/
│       ├── 06_super_learners/
│       └── submission_log.csv
│
└── 03_pres/
    ├── 00_analysis.R
    ├── figures/
    ├── tables/
    └── *.png  (figuras exportadas directamente)
```

---

# Pipeline de Replicación

El proyecto está organizado como un pipeline modular ejecutado desde:

```r
source("00_rundirectory.R")
```

El script maestro:

* carga paquetes,
* crea carpetas automáticamente,
* define rutas globales,
* ejecuta limpieza,
* genera features,
* construye folds espaciales,
* entrena modelos,
* genera submissions,
* exporta outputs finales.

---

# Flujo General del Proyecto

```text
Datos crudos
    ↓
00_clean.R
    ↓
01_text_variables.R
    ↓
02_spatial_variables.R
    ↓
03_imputation.R
    ↓
04_cv_setup.R
    ↓
Entrenamiento de modelos (00_linear_regression → 06_super_learning)
    ↓
03_pres/00_analysis.R
    ↓
Tablas y figuras finales
```

---

# Orden de Ejecución y Mapeo de Outputs

| Etapa | Script | Propósito | Outputs principales |
|---|---|---|---|
| 1 | `00_data/01_process/00_clean.R` | Limpieza base de Properati | `train_model.rds`, `test_model.rds` |
| 2 | `00_data/01_process/01_text_variables.R` | Variables derivadas desde texto | Features NLP |
| 3 | `00_data/01_process/02_spatial_variables.R` | Variables espaciales OSM + joins | Variables espaciales |
| 4 | `00_data/01_process/03_imputation.R` | Imputación y flags de missing | `train_model.rds`, `test_model.rds` |
| 5 | `00_data/01_process/04_cv_setup.R` | Construcción de folds | `folds_spatial.rds`, `folds_std.rds`, `folds_norte.rds`, `folds_spatial_upz.rds` |
| 6 | `02_models/00_training/00_linear_regression.R` | Benchmarks lineales | submissions OLS |
| 7 | `02_models/00_training/01_elastic_net.R` | Elastic Net | submissions EN |
| 8 | `02_models/00_training/02_regression_trees.R` | Árboles CART | submissions CART |
| 9 | `02_models/00_training/03_random_forest.R` | Random Forest | submissions RF |
| 10 | `02_models/00_training/04_boosting.R` | XGBoost + Bayesian Optimization | submissions XGB |
| 11 | `02_models/00_training/05_neural_network.R` | Redes neuronales (brulee/torch) | submissions NN |
| 12 | `02_models/00_training/06_super_learning.R` | Super Learner / Stacking | ensemble submissions |
| 13 | `03_pres/00_analysis.R` | Análisis profundo del mejor modelo | figuras y tablas finales |

---

# Outputs Generados

## Modelos entrenados

Ubicación:

```text
02_models/00_classes/
```

Incluye:

* workflows entrenados,
* tune_results,
* resample_results,
* hiperparámetros óptimos,
* objetos XGBoost,
* resultados resumidos.

---

## Submissions Kaggle

Ubicación:

```text
02_models/01_submissions/
```

Organizadas por familia de modelos:

* regresiones lineales,
* elastic net,
* CART,
* random forest,
* boosting,
* neural networks,
* super learners.

---

# Figuras Exportadas

Ubicación:

```text
03_pres/figures/
```

Ubicación: `03_pres/`

| Figura | Descripción |
|---|---|
| `benchmark_models.png` | Comparación de desempeño entre modelos |
| `spatial_gap.png` | Gap entre CV estándar y espacial |
| `feature_group_importance.png` | Importancia por grupos de features |
| `shap_importance.png` | Importancia SHAP |
| `shap_directionality.png` | Direccionalidad de variables SHAP |
| `shap_summary_top25.png` | Top 25 variables SHAP |
| `bias_by_estrato.png` | Sesgo por estrato |
| `bias_by_localidad_top15.png` | Error territorial por localidad |
| `map_error_spatial.png` | Mapa espacial del error |
| `map_error_norte.png` | Mapa de error — holdout norte |
| `map_error_upz.png` | Mapa de error por UPZ |
| `error_vs_price.png` | Error vs precio observado |
| `catastrophic_error_summary.png` | Resumen de errores críticos |
| `calibration_plot.png` | Diagnóstico de calibración |
| `xgb_gain_importance.png` | Gain importance XGBoost |
| `tabla_cv_performance.png` | Tabla de performance (imagen) |
| `tabla_cv_risk.png` | Tabla de riesgo (imagen) |
| `tabla_gap_spatial.png` | Tabla gap espacial (imagen) |
| `tabla_mae_folds.png` | Tabla MAE por fold (imagen) |

---

# Tablas Exportadas

Ubicación: `03_pres/` y `03_pres/tables/`

| Tabla | Contenido |
|---|---|
| `benchmark_models.csv` | Métricas de benchmark por modelo |
| `tabla_cv_performance.csv` | Performance general de modelos |
| `tabla_cv_risk.csv` | Riesgo de overprediction |
| `tabla_gap_spatial.csv` | Gap entre CV random y espacial |
| `tabla_mae_folds.csv` | MAE por fold |
| `feature_group_importance.csv` | Importancia por grupos |
| `risk_analysis.csv` | Evaluación económica del error |
| `shap_summary.csv` | Variables SHAP |
| `top_100_catastrophic_errors.csv` | Mayores errores del modelo |

---

# Datos

## Fuente

Kaggle — `uniandes-bdml-2026-10-ps3`

Datos de Properati para Bogotá D.C.

| Dataset | Observaciones |
|---|---|
| Train | 38,644 |
| Test | 10,286 |

Período aproximado:

* 2019–2021

---

# Variable Objetivo

```text
price
```

Precio de oferta del inmueble en pesos colombianos.

---

# Variables Utilizadas

## 1. Variables estructurales

Características físicas del inmueble:

* superficie,
* habitaciones,
* baños,
* tipo de propiedad,
* coordenadas,
* área cubierta.

---

## 2. Variables derivadas de texto

Construidas usando NLP básico sobre descripciones.

Ejemplos:

* terraza,
* ascensor,
* vigilancia,
* remodelado,
* duplex,
* chimenea,
* walk-in closet.

---

## 3. Variables espaciales

Construidas desde OpenStreetMap.

Incluyen proximidad o densidad de:

* hospitales,
* universidades,
* restaurantes,
* parques,
* cafés,
* bancos,
* supermercados,
* gimnasios,
* estaciones.

---

## 4. Variables administrativas

Construidas mediante joins espaciales:

* localidad,
* UPZ,
* estrato.

---

## 5. Missingness Flags

Indicadores binarios de imputación:

* `surface_missing`
* `rooms_missing`
* `upz_missing`
* `stratum_missing`

---

# Metodología

Se comparan múltiples modelos de regresión usando MAE como métrica principal.

| Modelo | Librería | Tipo |
|---|---|---|
| OLS | stats | Benchmark |
| Elastic Net | glmnet | Regularización |
| CART | rpart | Árbol |
| Random Forest | ranger | Ensemble |
| XGBoost | xgboost | Boosting |
| Neural Network | brulee | Deep learning |
| Super Learner | custom ensemble | Stacking |

---

# Validación Cruzada

## CV estándar

* 5-fold random cross-validation

## CV espacial

Cuatro esquemas implementados en `00_data/02_cv/`:

* `folds_spatial.rds` — spatial block CV por localidad (`group_vfold_cv`)
* `folds_spatial_upz.rds` — spatial block CV por UPZ (`group_vfold_cv`)
* `folds_norte.rds` — holdout norte manual (`manual_rset`)
* `pesos_dist_6km.rds` / `pesos_uniform.rds` — pesos de distancia para CV ponderado

Objetivo: medir capacidad real de generalización espacial hacia Chapinero.

---

# Mejor Modelo

El mejor desempeño fue obtenido por un **Super Learner (NNLS)** compuesto por:

* XGBoost
* Elastic Net

Sin embargo, el metalearner asignó:

* Peso 1.00 → XGBoost
* Peso 0.00 → Elastic Net

Por lo tanto, el desempeño final converge efectivamente a un modelo XGBoost.

---

# Hiperparámetros Óptimos

```r
nrounds          = 1000
max_depth        = 6
eta              = 0.05
subsample        = 0.8
min_child_weight = 10
```

---

# Resultados

## Performance principal

* Kaggle Public MAE ≈ COP $183.9M
* Top 3 leaderboard público
* Reducción ≈ 70% frente a benchmark naive

---

# Hallazgos Principales

* Variables OSM y texto aportan aproximadamente 35% del poder predictivo
* El contexto urbano importa tanto como las características físicas
* Bathrooms es la variable más importante
* El modelo se calibra bien en rangos medios
* El modelo tiende a subestimar propiedades premium

---

# Validación Espacial

El northeast holdout presenta un deterioro aproximado de:

```text
+7.6% MAE
```

respecto al CV aleatorio.


---

# Software

* R ≥ 4.3.0

---

# Paquetes Requeridos

Instalados automáticamente mediante `pacman::p_load()`.

| Categoría | Paquetes |
|---|---|
| Datos | tidyverse, janitor, skimr |
| Spatial | sf, osmdata, spatialsample |
| ML | caret, glmnet, ranger, xgboost, lightgbm, tidymodels |
| Texto | tokenizers, stopwords, SnowballC |
| Métricas | yardstick, MLmetrics |
| Visualización | ggplot2, patchwork |
| Tablas | gt, gtsummary |

Si `pacman` no está instalado:

```r
install.packages("pacman")
```

---

# Referencias

* Rosen, S. (1974). *Hedonic Prices and Implicit Markets*. Journal of Political Economy.
* Zillow Offers Case (2021).
* OpenStreetMap Contributors.
* Sarmiento-Barbieri, I. (2026). *Big Data and Machine Learning for Applied Economics*. Universidad de los Andes.
* Properati Colombia Dataset.

---

# Contacto

Para preguntas relacionadas con este repositorio, contactar a cualquiera de los autores listados anteriormente.
