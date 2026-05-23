# ============================================================
# 02_cart.R
# ============================================================

# 1. CV ESPACIAL ---------------------------------------------
train_sf <- st_as_sf(train, coords = c("lon", "lat"), crs = 4326, remove = FALSE) |>
  st_transform(crs = 3116)

set.seed(SEED)

cv_folds <- spatial_block_cv(
  train_sf,
  v = CV_FOLDS,
  cellsize = 2000
)

# 2. RECIPE --------------------------------------------------
recipe_cart <- recipe(log_price ~ ., data = train) |>
  step_rm(property_id, description, title, price) |>
  step_mutate(property_type = as.factor(property_type)) |>
  step_impute_median(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors())

# 3. SPEC + WORKFLOW -----------------------------------------
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

# 4. GRID DE TUNING ------------------------------------------
grid_cart <- grid_regular(
  cost_complexity(range = c(-4, -1)),
  tree_depth(range      = c(3, 10)),
  min_n(range           = c(5, 30)),
  levels = 4
)

# 5. CV ------------------------------------------------------
cv_results_cart <- tune_grid(
  wf_cart,
  resamples = cv_folds,
  grid      = grid_cart,
  metrics   = metric_set(mae)
)

collect_metrics(cv_results_cart) |>
  arrange(mean) |>
  head(10)

# 6. MODELO FINAL --------------------------------------------
best_cart <- select_best(
  cv_results_cart,
  metric = "mae"
)

modelo_cart <- wf_cart |>
  finalize_workflow(best_cart) |>
  fit(data = train)

# 7. LOG DEL MODELO ------------------------------------------
nombre_cart <- "CART_tuned"

log_modelo(
  cv_results_cart,
  nombre_cart
)

# 8. PREDICCIÓN SOBRE TEST -----------------------------------

pred_cart_log <- predict(
  modelo_cart,
  new_data = test
) |>
  pull(.pred)

pred_cart_price <- exp(pred_cart_log) - 1

# Evitar valores negativos o absurdos
pred_cart_price <- pmax(pred_cart_price, 0)

# 9. CREAR SUBMISSION ----------------------------------------

submission_cart <- tibble(
  property_id = test$property_id,
  price = pred_cart_price
)

# 10. GUARDAR CSV EN CARPETA CART ----------------------------

dir.create(
  here::here("02_models", "01_submissions", "02_regression_trees"),
  recursive = TRUE,
  showWarnings = FALSE
)

write_csv(
  submission_cart,
  here::here(
    "02_models",
    "01_submissions",
    "02_regression_trees",
    "CART_cart.csv"
  )
)

message("Submission CART guardada en: 02_models/01_submissions/02_regression_trees/CART_cart.csv")
