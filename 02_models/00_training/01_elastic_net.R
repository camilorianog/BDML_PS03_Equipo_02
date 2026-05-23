# ============================================================
# 01_elastic_net.R
# ============================================================

# 1. CV ESPACIAL ---------------------------------------------
train_sf <- st_as_sf(train, coords = c("lon", "lat"), crs = 4326, remove = FALSE) |>
  st_transform(crs = 3116)

set.seed(SEED)
cv_folds <- spatial_block_cv(train_sf, v = CV_FOLDS, cellsize = 2000)

# 2. RECIPE --------------------------------------------------
# Elastic Net requiere normalización y no tolera NAs
recipe_en <- recipe(log_price ~ ., data = train) |>
  step_rm(property_id, description, title, price) |>
  step_mutate(property_type = as.factor(property_type)) |>
  step_impute_median(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())   # ← obligatorio para glmnet

# 3. SPEC + WORKFLOW -----------------------------------------
spec_en <- linear_reg(
  penalty = tune(),
  mixture = tune()   # 0 = Ridge, 1 = Lasso, medio = Elastic Net
) |>
  set_engine("glmnet") |>
  set_mode("regression")

wf_en <- workflow() |>
  add_recipe(recipe_en) |>
  add_model(spec_en)

# 4. GRID DE TUNING ------------------------------------------
grid_en <- grid_regular(
  penalty(range = c(-4, 1)),   # 10^-4 a 10^1
  mixture(range = c(0, 1)),
  levels = 10
)

# 5. CV ------------------------------------------------------
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

# 6. MODELO FINAL + LOG + SUBMISSION -------------------------
best_en <- select_best(cv_results_en, metric = "mae")

modelo_en <- wf_en |>
  finalize_workflow(best_en) |>
  fit(train)

nombre_en <- "ElasticNet_tuned"
log_modelo(cv_results_en, nombre_en)
generar_submission(modelo_en, nombre_en)