# ============================================================
# 05_models_linear.R
# OLS (Regresión Lineal) y Elastic Net con tidymodels
# ============================================================
# Team 02 — Problem Set 03
#
# Modelos:
#   OLS_1 — solo variables estructurales Properati
#   OLS_2 — + variables de texto (01_text_variables.R)
#   OLS_3 — + variables espaciales OSM (especificación completa)
#   ENET  — Elastic Net, misma spec que OLS_3
#           tune sobre mixture (alpha) y penalty (lambda)
#
#
# CV: espacial (folds_spatial) como métrica principal
# Métrica: MAE sobre log_price
# ============================================================

p_load(tidymodels, glmnet)

# --- Cargar ------------------------------------------------------------------

train_sf      <- readRDS(here(paths$processed, "train_sf.rds"))
test          <- readRDS(here(paths$processed, "test_model.rds"))
folds_std     <- readRDS(here(paths$processed, "folds_std.rds"))
folds_spatial <- readRDS(here(paths$processed, "folds_spatial.rds"))

# --- Crear log_price ---------------------------------------------------------

train_sf <- train_sf |> mutate(log_price = log(price))

# Recrear folds con log_price incluido
set.seed(SEED)
folds_spatial <- group_vfold_cv(train_sf, group = LocCodigo)
set.seed(SEED)
folds_std     <- vfold_cv(train_sf, v = CV_FOLDS)

# --- Recipes -----------------------------------------------------------------

recipe_base <- recipe(
  log_price ~
    property_type +
    surface_total + surface_covered +
    rooms + bedrooms + bathrooms +
    ESTRATO +
    surface_total_was_na + surface_covered_was_na +
    rooms_was_na + bathrooms_was_na +
    ESTRATO_was_na,
  data = train_sf
) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors())

recipe_texto <- recipe(
  log_price ~
    property_type +
    surface_total + surface_covered +
    rooms + bedrooms + bathrooms +
    ESTRATO +
    surface_total_was_na + surface_covered_was_na +
    rooms_was_na + bathrooms_was_na + ESTRATO_was_na +
    parqueadero + parqueadero_inv +
    piscina + piscina_privada +
    ascensor + sin_ascensor +
    terraza + zona_exterior_priv +
    salon_comunal + sala_reuniones +
    porteria + vigilancia + conjunto +
    remodelada + antiguo + moderno +
    cocina_integral + cocina_americana +
    gym + chimenea + pisos_madera +
    walk_in_closet + lavanderia + duplex +
    apartaestudio + vista + vista_cerros +
    tenis + golf + squash +
    estudio_cuarto + escaleras_int,
  data = train_sf
) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors())

recipe_full <- recipe(
  log_price ~
    property_type +
    surface_total + surface_covered +
    rooms + bedrooms + bathrooms +
    ESTRATO + CODIGO_UPZ +
    EPT + AREA_HECTA +
    surface_total_was_na + surface_covered_was_na +
    rooms_was_na + bathrooms_was_na +
    ESTRATO_was_na + CODIGO_UPZ_was_na +
    parqueadero + parqueadero_inv +
    piscina + piscina_privada +
    ascensor + sin_ascensor +
    terraza + zona_exterior_priv +
    salon_comunal + sala_reuniones +
    porteria + vigilancia + conjunto +
    remodelada + antiguo + moderno +
    cocina_integral + cocina_americana +
    gym + chimenea + pisos_madera +
    walk_in_closet + lavanderia + duplex +
    apartaestudio + vista + vista_cerros +
    tenis + golf + squash +
    estudio_cuarto + escaleras_int +
    dist_cafe + dist_bus + dist_metro +
    dist_hospital + dist_colegio + dist_universidad +
    dist_parque + dist_super +
    n_cafes_500m + n_rest_500m + n_farmacias_500m +
    n_bancos_500m + n_gym_500m + n_lamparas_200m +
    is_residential,
  data = train_sf
) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors())

# Recipe full + normalización para Elastic Net
recipe_full_norm <- recipe_full |>
  step_normalize(all_numeric_predictors())

# --- Métrica -----------------------------------------------------------------

metricas <- metric_set(mae, rmse, rsq)

# --- Spec OLS ----------------------------------------------------------------

spec_lm <- linear_reg() |> set_engine("lm")

# =============================================================
# OLS_1 — base
# =============================================================

message("Entrenando OLS_1 (base Properati)...")
tic("OLS_1")

wf_ols1 <- workflow() |>
  add_recipe(recipe_base) |>
  add_model(spec_lm)

cv_ols1 <- fit_resamples(
  wf_ols1,
  resamples = folds_spatial,
  metrics   = metricas,
  control   = control_resamples(save_pred = TRUE)
)

toc(log = TRUE)
message("  MAE CV espacial OLS_1: ",
        round(collect_metrics(cv_ols1) |> filter(.metric == "mae") |> pull(mean), 4))

# =============================================================
# OLS_2 — + texto
# =============================================================

message("Entrenando OLS_2 (+ texto)...")
tic("OLS_2")

wf_ols2 <- workflow() |>
  add_recipe(recipe_texto) |>
  add_model(spec_lm)

cv_ols2 <- fit_resamples(
  wf_ols2,
  resamples = folds_spatial,
  metrics   = metricas,
  control   = control_resamples(save_pred = TRUE)
)

toc(log = TRUE)
message("  MAE CV espacial OLS_2: ",
        round(collect_metrics(cv_ols2) |> filter(.metric == "mae") |> pull(mean), 4))

# =============================================================
# OLS_3 — especificación completa
# =============================================================

message("Entrenando OLS_3 (completa)...")
tic("OLS_3")

wf_ols3 <- workflow() |>
  add_recipe(recipe_full) |>
  add_model(spec_lm)

cv_ols3 <- fit_resamples(
  wf_ols3,
  resamples = folds_spatial,
  metrics   = metricas,
  control   = control_resamples(save_pred = TRUE)
)

toc(log = TRUE)
message("  MAE CV espacial OLS_3: ",
        round(collect_metrics(cv_ols3) |> filter(.metric == "mae") |> pull(mean), 4))

# =============================================================
# ELASTIC NET
# =============================================================

message("Entrenando Elastic Net...")
tic("ENET")

spec_enet <- linear_reg(
  penalty = tune(),
  mixture = tune()
) |> set_engine("glmnet")

enet_grid <- grid_regular(
  penalty(range = c(-4, 1)),
  mixture(range = c(0, 1)),
  levels = c(25, 5)
)

wf_enet <- workflow() |>
  add_recipe(recipe_full_norm) |>
  add_model(spec_enet)

cv_enet <- tune_grid(
  wf_enet,
  resamples = folds_spatial,
  grid      = enet_grid,
  metrics   = metricas,
  control   = control_grid(save_pred = TRUE)
)

toc(log = TRUE)

best_enet <- select_best(cv_enet, metric = "mae")
message("  MAE CV espacial ENET: ",
        round(show_best(cv_enet, metric = "mae", n = 1)$mean, 4))
message("  Mejor penalty: ", round(best_enet$penalty, 5))
message("  Mejor mixture: ", best_enet$mixture)

# =============================================================
# RESUMEN COMPARATIVO
# =============================================================

resumen_lineal <- tibble(
  Modelo = c("OLS_1 (base)", "OLS_2 (+texto)", "OLS_3 (+OSM)", "Elastic Net"),
  MAE_CV_espacial = c(
    collect_metrics(cv_ols1) |> filter(.metric == "mae") |> pull(mean),
    collect_metrics(cv_ols2) |> filter(.metric == "mae") |> pull(mean),
    collect_metrics(cv_ols3) |> filter(.metric == "mae") |> pull(mean),
    show_best(cv_enet, metric = "mae", n = 1)$mean
  )
) |> arrange(MAE_CV_espacial)

message("\n--- Resumen modelos lineales ---")
print(resumen_lineal)

# =============================================================
# FITS FINALES — entrenar con TODOS los datos
# =============================================================

message("Entrenando modelos finales sobre todos los datos...")

fit_ols1 <- fit(wf_ols1, data = train_sf)
fit_ols2 <- fit(wf_ols2, data = train_sf)
fit_ols3 <- fit(wf_ols3, data = train_sf)

wf_enet_final <- finalize_workflow(wf_enet, best_enet)
fit_enet      <- fit(wf_enet_final, data = train_sf)

# =============================================================
# SUBMISSIONS
# =============================================================

generar_submission(fit_ols1, "LR_ols1")
generar_submission(fit_ols2, "LR_ols2")
generar_submission(fit_ols3, "LR_ols3")
generar_submission(fit_enet, "EN_enet")

# =============================================================
# GUARDAR
# =============================================================

saveRDS(fit_ols1,       here(paths$training, "fit_ols1.rds"))
saveRDS(fit_ols2,       here(paths$training, "fit_ols2.rds"))
saveRDS(fit_ols3,       here(paths$training, "fit_ols3.rds"))
saveRDS(fit_enet,       here(paths$training, "fit_enet.rds"))
saveRDS(cv_ols1,        here(paths$training, "cv_ols1.rds"))
saveRDS(cv_ols2,        here(paths$training, "cv_ols2.rds"))
saveRDS(cv_ols3,        here(paths$training, "cv_ols3.rds"))
saveRDS(cv_enet,        here(paths$training, "cv_enet.rds"))
saveRDS(resumen_lineal, here(paths$training, "resumen_lineal.rds"))

message("05_models_linear.R")