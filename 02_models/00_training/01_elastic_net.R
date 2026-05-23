# ============================================================
# 01_elastic_net.R
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
# Elastic Net requiere normalización y no tolera NAs

recipe_en <- recipe(log_price ~ ., data = train) |>
  step_rm(property_id, description, title, price,
          any_of(c("geometry", "shape"))) |>
  step_mutate(across(where(is.character), as.factor)) |>
  step_impute_median(all_numeric_predictors()) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_lincomb(all_numeric_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())   # ← obligatorio para glmnet

# 4. SPEC + WORKFLOW -----------------------------------------

spec_en <- linear_reg(
  penalty = tune(),
  mixture = tune()   # 0 = Ridge, 1 = Lasso, medio = Elastic Net
) |>
  set_engine("glmnet") |>
  set_mode("regression")

wf_en <- workflow() |>
  add_recipe(recipe_en) |>
  add_model(spec_en)

# 5. GRID DE TUNING ------------------------------------------

grid_en <- grid_regular(
  penalty(range = c(-4, 1)),   # 10^-4 a 10^1
  mixture(range = c(0, 1)),
  levels = 10
)

# 6. CV ------------------------------------------------------

cv_results_en <- tune_grid(
  wf_en,
  resamples = cv_folds,
  grid      = grid_en,
  metrics   = metric_set(mae)
)

collect_metrics(cv_results_en) |>
  arrange(mean) |>
  head(10)

autoplot(cv_results_en)   # ver penalización vs MAE

# 7. MODELO FINAL + LOG + SUBMISSION -------------------------

best_en <- select_best(cv_results_en, metric = "mae")

modelo_en <- wf_en |>
  finalize_workflow(best_en) |>
  fit(train)

nombre_en <- "EN_tuned"
log_modelo(cv_results_en, nombre_en)
generar_submission(modelo_en, nombre_en)