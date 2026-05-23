
# ============================================================
# 00_linear_regression.R
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

recipe_lr <- recipe(log_price ~ ., data = train) |>
  step_rm(property_id, description, title, price,
          any_of(c("geometry", "shape"))) |>
  step_mutate(across(where(is.character), as.factor)) |>
  step_impute_median(all_numeric_predictors()) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_lincomb(all_numeric_predictors()) |>
  step_zv(all_predictors())

# 4. SPEC + WORKFLOW -----------------------------------------

wf_lr <- workflow() |>
  add_recipe(recipe_lr) |>
  add_model(
    linear_reg() |> set_engine("lm") |> set_mode("regression")
  )

# 5. CV ------------------------------------------------------

cv_results_lr <- fit_resamples(
  wf_lr,
  resamples = cv_folds,
  metrics   = metric_set(mae)
)

collect_metrics(cv_results_lr)

# 6. MODELO FINAL + LOG + SUBMISSION -------------------------

modelo_lr <- wf_lr |> fit(train)
nombre_lr <- "LR_base"

log_modelo(cv_results_lr, nombre_lr)
generar_submission(modelo_lr, nombre_lr)
