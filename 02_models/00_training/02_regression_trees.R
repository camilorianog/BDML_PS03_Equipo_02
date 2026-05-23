# ============================================================
# 02_regression_trees.R
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

recipe_cart <- recipe(log_price ~ ., data = train) |>
  step_rm(property_id, description, title, price,
          any_of(c("geometry", "shape"))) |>
  step_mutate(across(where(is.character), as.factor)) |>
  step_impute_median(all_numeric_predictors()) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_lincomb(all_numeric_predictors()) |>
  step_zv(all_predictors())

# 4. SPEC + WORKFLOW -----------------------------------------

spec_cart <- decision_tree(
  cost_complexity = tune(),
  tree_depth      = tune(),
  min_n           = tune()
) |>
  set_engine("rpart") |>
  set_mode("regression")

wf_cart <- workflow() |>
  add_recipe(recipe_cart) |>
  add_model(spec_cart)

# 5. GRID DE TUNING ------------------------------------------

grid_cart <- grid_regular(
  cost_complexity(range = c(-4, -1)),
  tree_depth(range      = c(3, 10)),
  min_n(range           = c(5, 30)),
  levels = 4
)

# 6. CV ------------------------------------------------------

cv_results_cart <- tune_grid(
  wf_cart,
  resamples = cv_folds,
  grid      = grid_cart,
  metrics   = metric_set(mae)
)

collect_metrics(cv_results_cart) |>
  arrange(mean) |>
  head(10)

# 7. MODELO FINAL + LOG + SUBMISSION -------------------------

best_cart <- select_best(cv_results_cart, metric = "mae")

modelo_cart <- wf_cart |>
  finalize_workflow(best_cart) |>
  fit(train)

nombre_cart <- "CART_tuned"
log_modelo(cv_results_cart, nombre_cart)
generar_submission(modelo_cart, nombre_cart)