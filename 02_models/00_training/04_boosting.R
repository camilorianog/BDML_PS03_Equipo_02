# ============================================================
# 04_boosting.R
# XGBoost — tidymodels + finetune::tune_bayes()
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

recipe_xgb <- recipe(log_price ~ ., data = train) |>
  step_rm(property_id, description, title, price,
          any_of(c("geometry", "shape"))) |>
  step_mutate(across(where(is.character), as.factor)) |>
  step_impute_median(all_numeric_predictors()) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_lincomb(all_numeric_predictors()) |>
  step_zv(all_predictors())

# 4. SPEC + WORKFLOW -----------------------------------------

spec_xgb <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  min_n          = tune(),
  loss_reduction = tune(),
  sample_size    = tune(),
  mtry           = tune()
) |>
  set_engine("xgboost",
             nthread   = max(1L, parallel::detectCores() - 1L),
             objective = "reg:squarederror",
             eval_metric = "mae") |>
  set_mode("regression")

wf_xgb <- workflow() |>
  add_recipe(recipe_xgb) |>
  add_model(spec_xgb)

# 5. BAYESIAN OPTIMIZATION -----------------------------------
# finetune::tune_bayes() — minimiza MAE en spatial CV

set.seed(SEED)
cv_results_xgb <- finetune::tune_bayes(
  wf_xgb,
  resamples  = cv_folds,
  iter       = 30,
  initial    = 10,
  metrics    = metric_set(mae),
  param_info = parameters(
    trees(range          = c(300L, 1500L)),
    tree_depth(range     = c(3L,   8L)),
    learn_rate(range     = c(-3,   -1)),      # log10: 0.001–0.1
    min_n(range          = c(5L,   30L)),
    loss_reduction(range = c(-4,   2)),        # log10
    sample_size(range    = c(0.60, 0.90)),
    mtry(range           = c(5L,   40L))
  ),
  control = finetune::control_bayes(
    no_improve  = 10,
    verbose     = TRUE,
    save_pred   = FALSE
  )
)

collect_metrics(cv_results_xgb) |>
  arrange(mean) |>
  head(10)

autoplot(cv_results_xgb)

# 6. MODELO FINAL + LOG + SUBMISSION -------------------------

best_xgb <- select_best(cv_results_xgb, metric = "mae")

modelo_xgb <- wf_xgb |>
  finalize_workflow(best_xgb) |>
  fit(train)

nombre_xgb <- nm("XGB", best_xgb)
log_modelo(cv_results_xgb, nombre_xgb)
generar_submission(modelo_xgb, nombre_xgb)
