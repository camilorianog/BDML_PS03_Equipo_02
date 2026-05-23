# ============================================================
# 03_random_forest.R
# Random Forest — tidymodels + ranger engine
# ============================================================

# 1. DATOS ---------------------------------------------------

if (exists("train", envir = .GlobalEnv)) {
  train <- get("train", envir = .GlobalEnv)
} else {
  train <- readRDS(here(paths$processed, "train_model.rds"))
}
if (!"log_price" %in% names(train)) train$log_price <- log(train$price)

if (exists("test", envir = .GlobalEnv)) {
  test <- get("test", envir = .GlobalEnv)
} else {
  test <- readRDS(here(paths$processed, "test_model.rds"))
}

# 2. CV ESPACIAL ---------------------------------------------

if (exists("folds_spatial", envir = .GlobalEnv)) {
  cv_folds <- get("folds_spatial", envir = .GlobalEnv)
} else {
  cv_folds <- readRDS(here(paths$cv, "folds_spatial.rds"))
}

# 3. RECIPE --------------------------------------------------

recipe_rf <- recipe(log_price ~ ., data = train) |>
  step_rm(property_id, description, title, price,
          any_of(c("geometry", "shape"))) |>
  step_mutate(across(where(is.character), as.factor)) |>
  step_impute_median(all_numeric_predictors()) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_lincomb(all_numeric_predictors()) |>
  step_zv(all_predictors())

# 4. SPEC + WORKFLOW -----------------------------------------

spec_rf <- rand_forest(
  mtry  = tune(),
  trees = tune(),
  min_n = tune()
) |>
  set_engine("ranger",
             splitrule               = "variance",
             respect.unordered.factors = "order",
             num.threads             = max(1L, parallel::detectCores() - 1L),
             seed                    = SEED) |>
  set_mode("regression")

wf_rf <- workflow() |>
  add_recipe(recipe_rf) |>
  add_model(spec_rf)

# 5. GRID DE TUNING ------------------------------------------

grid_rf <- grid_regular(
  mtry(range  = c(5L, 30L)),
  trees(range = c(500L, 1000L)),
  min_n(range = c(3L, 15L)),
  levels = 4
)

# 6. CV ------------------------------------------------------

set.seed(SEED)
cv_results_rf <- tune_grid(
  wf_rf,
  resamples = cv_folds,
  grid      = grid_rf,
  metrics   = metric_set(mae)
)

collect_metrics(cv_results_rf) |>
  arrange(mean) |>
  head(10)

autoplot(cv_results_rf)

# 7. MODELO FINAL + LOG + SUBMISSION -------------------------

best_rf <- select_best(cv_results_rf, metric = "mae")

modelo_rf <- wf_rf |>
  finalize_workflow(best_rf) |>
  fit(train)

nombre_rf <- nm("RF", best_rf)
log_modelo(cv_results_rf, nombre_rf)
generar_submission(modelo_rf, nombre_rf)
