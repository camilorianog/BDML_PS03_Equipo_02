
# ============================================================
# 00_linear_regression.R
# ============================================================

# 1. CV ESPACIAL ---------------------------------------------

train_sf <- st_as_sf(train, coords = c("lon", "lat"), crs = 4326, remove = FALSE) |>
  st_transform(crs = 3116)

set.seed(SEED)
cv_folds <- spatial_block_cv(train_sf, v = CV_FOLDS, cellsize = 2000)

autoplot(cv_folds) + theme_minimal()  # verificar bloques geográficos

# 2. RECIPE --------------------------------------------------

recipe_lr <- recipe(log_price ~ ., data = train) |>
  step_rm(property_id, description, title, price) |>
  step_mutate(property_type = as.factor(property_type)) |>
  step_impute_median(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors())

# 3. SPEC + WORKFLOW -----------------------------------------

wf_lr <- workflow() |>
  add_recipe(recipe_lr) |>
  add_model(
    linear_reg() |> set_engine("lm") |> set_mode("regression")
  )

# 4. CV ------------------------------------------------------

cv_results_lr <- fit_resamples(
  wf_lr,
  resamples = cv_folds,
  metrics   = metric_set(mae)
)

collect_metrics(cv_results_lr)

# 5. MODELO FINAL + LOG + SUBMISSION -------------------------

modelo_lr <- wf_lr |> fit(train)
nombre_lr <- "LR_base"

log_modelo(cv_results_lr, nombre_lr)
generar_submission(modelo_lr, nombre_lr)
