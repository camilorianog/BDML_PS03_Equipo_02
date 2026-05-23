
# 00_rundirectory.R ------------------------------------------------------
# Universidad de Los Andes - 2026-10 - Big Data y Machine Learning para Economía Aplicada
# Problem Set 03: Making Money With ML?

# Team 02
#   · Jose A. Rincón S.  — 202013328
#   · Juan C. Riaño      — 202013305
#   · Lucas Rodriguez    — 202021985
#   · Santiago González  — 202110234

# Paquetes ----------------------------------------------------------------

if (!require("pacman", quietly = TRUE)) install.packages("pacman")

p_load(
  # Entorno
  here, tictoc, NoSleepR,
  
  # Manipulación de datos
  tidyverse, janitor, skimr, gt, gtsummary, sf, stringr,
  
  # Manipulación de texto
  tokenizers, stopwords, SnowballC, wordcloud, stringr, stringi,
  
  # Datos
  osmdata,
  
  # Modelado
  caret, glmnet, naivebayes, ranger, xgboost, lightgbm, bonsai, rlang, tidymodels, spatialsample, recipes, brulee, finetune,
  
  # Métricas
  yardstick, MLmetrics, 
  
  # Visualización
  ggplot2, patchwork
)

# Admin ------------------------------------------------------------------

## Parámetros Globales

SEED     <- 202601
CV_FOLDS <- 5

set.seed(SEED)

## Creación de carpetas ---------------------------------------------------

paths <- list(
  root        = here::here(),
  raw         = here("00_data", "00_raw"),
  competition = here("00_data", "00_raw","00_competition"),
  process     = here("00_data", "01_process"),
  processed   = here("00_data", "01_processed"),
  cv          = here("00_data", "02_cv"),
  functions   = here("01_R",    "00_functions"),
  models      = here("02_models"),
  training    = here("02_models", "00_training"),
  submissions = here("02_models", "01_submissions"),
  LR          = here("02_models", "01_submissions", "00_linear_regression"),
  EN          = here("02_models", "01_submissions", "01_elastic_net"),
  CART        = here("02_models", "01_submissions", "02_regression_trees"),
  RF          = here("02_models", "01_submissions", "03_random_forest"),
  Boosting    = here("02_models", "01_submissions", "04_boosting"),
  NN          = here("02_models", "01_submissions", "05_neural_networks"),
  Super       = here("02_models", "01_submissions", "06_super_learners"),
  figures     = here("03_pres", "figures"),
  tables      = here("03_pres", "tables")
)

invisible(lapply(paths, dir.create, recursive = TRUE, showWarnings = FALSE))


## funciones auxiliares --------------------------------------------------

source(here(paths$functions, "00_nm.R"))                    # Genera nombre del modelo
source(here(paths$functions, "01_log_modelo.R"))            # Registra en un log cada modelo con datos importantes
source(here(paths$functions, "02_generar_submission.R"))    # Genera el archivo csv con la submission 

# Pipeline ---------------------------------------------------------------


## 1. Limpieza de datos ------------------------------------------------------

tic("Limpieza")
source(here(paths$process, "00_clean.R"))
toc(log = TRUE)

tic("Texto a variables")
source(here(paths$process, "01_text_variables.R"))
toc(log = TRUE)

tic("Variables espaciales")
source(here(paths$process, "02_spatial_variables.R"))
toc(log = TRUE)

tic("Imputación")
source(here(paths$process, "03_imputation.R"))
toc(log = TRUE)

## 3. CV (tidymodels) --------------------------------------------------------

tic("CV setup")
source(here(paths$process, "04_cv_setup.R"))
toc(log = TRUE)

## 4. Modelado ---------------------------------------------------------------

tic("0. Regresión lineal")
source(here(paths$training, "00_linear_regression.R"))
toc(log = TRUE)

tic("1. Elastic Net")
source(here(paths$training, "01_elastic_net.R"))
toc(log = TRUE)

tic("2. Regression Trees")
source(here(paths$training, "02_regression_trees.R"))
toc(log = TRUE)

tic("3. Random Forest")
source(here(paths$training, "03_random_forest.R"))
toc(log = TRUE)

tic("4. Boost")
source(here(paths$training, "04_boosting.R"))
toc(log = TRUE)

tic("5. Red neuronal")
source(here(paths$training, "05_neural_network.R"))
toc(log = TRUE)

tic("6. Super Learner / Stacking")
source(here(paths$training, "06_super_learning.R"))
toc(log = TRUE)

## 5. Analisis presentación --------------------------------------------------

tic("7. Análisis presentación")
source(here("03_pres", "00_analysis.R"))
toc(log = TRUE)
