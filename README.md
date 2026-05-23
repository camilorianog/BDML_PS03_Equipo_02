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

Los datos crudos deben descargarse manualmente desde Kaggle antes de ejecutar el pipeline y desde fuentes externas acá registradas.

```r
# 1. Descargar datos desde Kaggle
# kaggle competitions download -c uniandes-bdml-2026-10-ps3
## Datos Externos Geoespaciales

Además de los datos originales de Properati, el proyecto incorpora múltiples capas geoespaciales externas provenientes de Datos Abiertos Bogotá.
Estas capas permiten construir variables espaciales y administrativas críticas para mejorar la capacidad de generalización del modelo hacia Chapinero.
Se dejaron cargadas en raw data los datos de UPZ y de Localidades. El de manzanas por lo pesado se deja claro el proceso de descarga.

# 2.1. Estratificación Socioeconómica

Fuente oficial:

[Datos Abiertos Bogotá — Estratificación para Bogotá](https://datosabiertos.bogota.gov.co/dataset/estratificacion-para-bogota?)

Archivo utilizado:

ManzanaEstratificacion.shp

## 2.2. UPZ (Unidades de Planeamiento Zonal)

Fuente oficial:

[Datos Abiertos Bogotá — Espacio Público Total por UPZ 2021](https://datosabiertos.bogota.gov.co/en/dataset/espacio-publico-total-upz-2021?)

Archivo utilizado:

EPT_UPZ.shp


### 3. Localidades de Bogotá D.C.

Fuente oficial:

[Datos Abiertos Bogotá — Localidades Bogotá D.C.](https://datosabiertos.bogota.gov.co/dataset/localidad-bogota-d-c?)

Archivo utilizado:

Loca.shp

# 2. Los shapefiles deben descargarse manualmente y almacenarse dentro de:

00_data/00_raw/


# 3. Ejecutar pipeline maestro
source("00_rundirectory.R")
```

El script maestro ejecuta secuencialmente:

1. Limpieza y preparación de datos
2. Construcción de variables espaciales y textuales
3. Feature engineering
4. Construcción de matrices de modelado
5. Entrenamiento de modelos benchmark y avanzados
6. Validación cruzada estándar y espacial
7. Generación de submissions Kaggle
8. Producción de tablas y figuras para presentación final

---

# Estructura del Repositorio

```text
.
├── 00_rundirectory.R                # Script maestro — punto de entrada
├── BDML_PS03_Equipo_02.Rproj        # Proyecto de RStudio
│
├── 00_data/
│   ├── 00_raw/                      # Datos crudos + shapefiles + OSM cache
│   └── 01_processed/                # Datos procesados y matrices finales
│
├── 01_R/
│   ├── 00_prep/
│   │   ├── 00_clean.R               # Limpieza de datos Properati
│   │   ├── 01_eda.R                 # Exploratory Data Analysis
│   │   ├── 02_spatial_join.R        # Join espacial UPZ/localidad
│   │   └── 03_text_cleaning.R       # Limpieza de texto
│   │
│   ├── 01_feat/
│   │   ├── 00_text_features.R       # Variables desde descripción
│   │   ├── 01_osm_features.R        # Variables espaciales OSM
│   │   ├── 02_feature_engineering.R # Variables derivadas
│   │   └── 03_missing_flags.R       # Indicadores de missings
│   │
│   ├── 02_functions/
│   │   ├── 00_cv_spatial.R          # Spatial cross-validation
│   │   ├── 01_model_metrics.R       # Métricas y evaluación
│   │   ├── 02_save_model.R          # Guardado de modelos
│   │   └── 03_submission.R          # Generación submissions Kaggle
│   │
│   └── 03_validation/
│       ├── 00_spatial_gap.R         # Comparación CV espacial vs random
│       └── 01_calibration.R         # Diagnóstico de calibración
│
├── 02_models/
│   ├── 00_training/
│   │   ├── 01_OLS.R
│   │   ├── 02_ElasticNet.R
│   │   ├── 03_RandomForest.R
│   │   ├── 04_XGBoost.R
│   │   └── 05_SuperLearner.R
│   │
│   └── 01_submissions/
│       ├── 00_linear_regression/
│       ├── 01_elastic_net/
│       ├── 02_xgboost/
│       ├── 03_random_forest/
│       └── 04_boosting/
│
├── 03_pres/
│   ├── figures/                     # Figuras finales del deck
│   └── tables/                      # Tablas exportadas
│
└── README.md
```

---

# Datos

## Fuente

Kaggle — `uniandes-bdml-2026-10-ps3`

Los datos provienen de Properati e incluyen anuncios de vivienda para Bogotá D.C.

| Dataset | Observaciones |
| ------- | ------------- |
| Train   | 38,644        |
| Test    | 10,286        |

Período aproximado:

* 2019–2021

Composición:

* 97.3% apartamentos
* 2.7% casas

---

# Variable Objetivo

```text
price
```

Precio de oferta del inmueble en pesos colombianos (COP).

---

# Variables Utilizadas

## 1. Características estructurales

Determinantes físicos directos del precio:

* surface_total
* surface_covered
* rooms
* bedrooms
* bathrooms
* property_type
* latitud / longitud

---

## 2. Variables derivadas de texto

Se construyen utilizando procesamiento simple de texto sobre la descripción del inmueble.

Ejemplos:

* ascensor
* terraza
* piscina
* vigilancia
* remodelado
* chimenea
* walk-in closet
* vista
* duplex

Estas variables permiten capturar calidad y amenidades no observables en variables estructurales tradicionales.

---

# Justificación Económica

Siguiendo a Rosen (1974), el precio de la vivienda puede representarse como:

P = f(estructura,localizaci\acute{o}n,amenidades)

Las descripciones de los anuncios contienen señales relevantes sobre calidad, seguridad y amenities que afectan la disposición a pagar.

---

## 3. Variables espaciales (OSM)

Construidas desde OpenStreetMap:

### Distancias

* cafés
* hospitales
* parques
* universidades
* estaciones de transporte
* supermercados
* colegios

### Densidades en buffers

* restaurantes
* bancos
* gimnasios
* farmacias

Estas variables aproximan accesibilidad urbana y calidad del entorno.

---

## 4. Variables administrativas

* localidad
* UPZ
* estrato
* proxies espaciales

Capturan la fuerte segmentación territorial de Bogotá.

---

## 5. Missingness Flags

Indicadores binarios del proceso de imputación:

* surface_missing
* rooms_missing
* upz_missing
* stratum_missing

Los patrones de missing pueden contener información económica relevante sobre calidad del anuncio y segmentación de mercado.

---

# Metodología

Se comparan múltiples modelos de regresión utilizando MAE como métrica principal.

| # | Modelo        | Librería     | Notas                |
| - | ------------- | ------------ | -------------------- |
| 1 | OLS           | stats        | Benchmark lineal     |
| 2 | Elastic Net   | glmnet       | Regularización L1/L2 |
| 3 | Random Forest | ranger       | Ensamble de árboles  |
| 4 | XGBoost       | xgboost      | Gradient boosting    |
| 5 | Super Learner | SuperLearner | Ensemble final       |

---

# Validación Cruzada

## Validación estándar

* 5-fold random CV

## Validación espacial

* Leave-location-out CV
* Holdout espacial northeast
* Evaluación por localidad / UPZ

El objetivo es medir capacidad real de generalización espacial hacia Chapinero.

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

## Hiperparámetros óptimos

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
* El contexto urbano importa tanto como las características físicas del inmueble
* Bathrooms es la variable individual más importante
* El modelo presenta buena calibración en rangos medios
* En propiedades > COP $1.2B tiende a subestimar

---

# Validación Espacial

El northeast holdout presenta un deterioro aproximado de:

```text
+7.6% MAE
```

respecto al CV aleatorio.

Esto sugiere que:

* El modelo generaliza razonablemente bien dentro de Bogotá
* El riesgo aumenta en zonas premium similares a Chapinero
* El error real podría ser mayor al estimado por CV estándar

---

# Software

* R ≥ 4.3.0

---

# Paquetes Requeridos

Instalados automáticamente mediante `pacman::p_load()`.

| Categoría     | Paquetes                                     |
| ------------- | -------------------------------------------- |
| Datos         | tidyverse, data.table, janitor               |
| Spatial       | sf, terra, osmdata, tmap                     |
| ML            | caret, glmnet, ranger, xgboost, SuperLearner |
| Texto         | tidytext, quanteda, stringr                  |
| Métricas      | yardstick, MLmetrics                         |
| Visualización | ggplot2, patchwork, viridis                  |
| Tablas        | gt, kableExtra                               |

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
