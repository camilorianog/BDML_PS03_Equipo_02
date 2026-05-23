# ============================================================
# 05_neural_network.R
# Red neuronal — tidymodels + brulee (torch)
# ============================================================
# Requisitos (sesión interactiva o 00_rundirectory.R):
#   - train, test en memoria (03_imputation.R o train_model.rds)
#   - folds_* en memoria (04_cv_setup.R o folds_*.rds)
#   - paquetes: brulee, torch (instalados vía 00_rundirectory.R)
# ============================================================

# --- Hiperparámetros (ajustar aquí) ----------------------------------------
#
# hidden_units : vector de neuronas por capa oculta.
#                c(10) = una capa; c(32, 16) = dos capas.
# penalty      : weight decay L2.
# learn_rate   : tasa de aprendizaje del optimizador.
# epochs       : épocas de entrenamiento.
# activation   : "relu", "elu", "selu", "sigmoid", "tanh".
# dropout      : fracción de dropout (0 = desactivado).
# cv_set       : folds de 04_cv_setup.R:
#                  "folds_std"         — 5-fold aleatorio
#                  "folds_global"      — estratificado por localidad
#                  "folds_spatial"     — leave-location-out
#                  "folds_spatial_upz" — leave-UPZ-out

NN_HIDDEN     <- c(128, 64, 32)
NN_PENALTY    <- 0.001
NN_LEARN_RATE <- 0.001
NN_EPOCHS     <- 300
NN_ACTIVATION <- "relu"
NN_DROPOUT    <- 0
CV_SET        <- "folds_spatial"

# --- Datos -------------------------------------------------------------------

prepare_modeling_tbl <- function(x) {
  if (inherits(x, "sf")) {
    x <- sf::st_drop_geometry(x)
  }
  as.data.frame(x)
}

if (exists("train", envir = .GlobalEnv)) {
  train <- prepare_modeling_tbl(get("train", envir = .GlobalEnv))
} else if (file.exists(here(paths$processed, "train_model.rds"))) {
  train <- prepare_modeling_tbl(readRDS(here(paths$processed, "train_model.rds")))
} else {
  stop(
    "No hay objeto 'train' en memoria ni train_model.rds. ",
    "Ejecuta 03_imputation.R o el pipeline desde 00_rundirectory.R.",
    call. = FALSE
  )
}

if (!"log_price" %in% names(train)) {
  train$log_price <- log(train$price)
}


if (exists("test", envir = .GlobalEnv)) {
  test <- prepare_modeling_tbl(get("test", envir = .GlobalEnv))
} else if (file.exists(here(paths$processed, "test_model.rds"))) {
  test <- prepare_modeling_tbl(readRDS(here(paths$processed, "test_model.rds")))
} else {
  stop(
    "No hay objeto 'test' en memoria ni test_model.rds. ",
    "Ejecuta 03_imputation.R o el pipeline desde 00_rundirectory.R.",
    call. = FALSE
  )
}

valid_cv_sets <- c(
  "folds_std", "folds_global", "folds_spatial", "folds_spatial_upz"
)
if (!CV_SET %in% valid_cv_sets) {
  stop(
    "CV_SET debe ser uno de: ",
    paste(valid_cv_sets, collapse = ", "),
    call. = FALSE
  )
}

if (exists(CV_SET, envir = .GlobalEnv)) {
  cv_folds <- get(CV_SET, envir = .GlobalEnv)
} else {
  cv_path <- here(paths$cv, paste0(CV_SET, ".rds"))
  if (!file.exists(cv_path)) {
    stop(
      "No hay '", CV_SET, "' en memoria ni ", CV_SET, ".rds. ",
      "Ejecuta 04_cv_setup.R primero.",
      call. = FALSE
    )
  }
  cv_folds <- readRDS(cv_path)
}

message(
  "NN (brulee) | CV: ", CV_SET,
  " | hidden=", paste(NN_HIDDEN, collapse = "-"),
  " | penalty=", NN_PENALTY,
  " | learn_rate=", NN_LEARN_RATE,
  " | epochs=", NN_EPOCHS
)

# --- Recipe ------------------------------------------------------------------

recipe_nn <- recipe(log_price ~ ., data = train) |>
  step_rm(
    property_id, description, title, price,
    any_of(c("geometry", "shape"))
  ) |>
  step_mutate(
    across(where(is.integer),   as.numeric),
    across(where(is.character), as.factor)
  ) |>
  step_impute_median(all_numeric_predictors()) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_lincomb(all_numeric_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

# --- Modelo + workflow -------------------------------------------------------

spec_nn <- mlp(
  hidden_units = NN_HIDDEN,
  penalty      = NN_PENALTY,
  learn_rate   = NN_LEARN_RATE,
  epochs       = NN_EPOCHS,
  activation   = NN_ACTIVATION,
  dropout      = NN_DROPOUT
) |>
  set_engine("brulee") |>
  set_mode("regression")

wf_nn <- workflow() |>
  add_recipe(recipe_nn) |>
  add_model(spec_nn)

# --- Prevenir bloqueo de pantalla --------------------------------------------

NoSleepR::nosleep_on()
on.exit(NoSleepR::nosleep_off(), add = TRUE)

# --- Validación cruzada -------------------------------------------------------

set.seed(SEED)

cv_results_nn <- fit_resamples(
  wf_nn,
  resamples = cv_folds,
  metrics   = metric_set(mae, rmse, rsq)
)

collect_metrics(cv_results_nn)

# --- Modelo final + log + submission -----------------------------------------

modelo_nn <- wf_nn |> fit(train)

# Nombre generado automáticamente desde los hiperparámetros actuales
nombre_nn <- nm("BRU", tibble(
  hidden   = paste(NN_HIDDEN, collapse = "-"),
  penalty  = NN_PENALTY,
  lr       = NN_LEARN_RATE,
  epochs   = NN_EPOCHS,
  act      = NN_ACTIVATION,
  drop     = NN_DROPOUT,
  cv       = gsub("folds_", "", CV_SET)
))

log_modelo(cv_results_nn, nombre_nn)
generar_submission(modelo_nn, nombre_nn, log_scale = FALSE)

# ============================================================
# OPCIONAL: tune_grid
# ============================================================
#
# spec_nn_tune <- mlp() |>
#   set_engine("brulee") |>
#   set_mode("regression")
#
# wf_nn_tune <- workflow() |>
#   add_recipe(recipe_nn) |>
#   add_model(spec_nn_tune)
#
# grid_nn <- expand_grid(
#   hidden_units = list(c(16), c(32, 16), c(64, 32, 16)),
#   penalty      = 10^runif(5, -4, -2),
#   learn_rate   = 10^runif(5, -3, -1),
#   epochs       = sample(c(50, 100, 150), 5, replace = TRUE),
#   activation   = "relu",
#   dropout      = c(0, 0.1, 0.2)
# )
#
# set.seed(SEED)
# tuned_nn <- tune_grid(
#   wf_nn_tune,
#   resamples = cv_folds,
#   grid      = grid_nn,
#   metrics   = metric_set(mae)
# )
#
# best_nn <- select_best(tuned_nn, metric = "mae")
# wf_nn_final <- finalize_workflow(wf_nn_tune, best_nn)
# modelo_nn <- fit(wf_nn_final, train)
# nombre_nn <- nm("BRU", best_nn)
# log_modelo(tuned_nn, nombre_nn, best = best_nn)
# generar_submission(modelo_nn, nombre_nn)
