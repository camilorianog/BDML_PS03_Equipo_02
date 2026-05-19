# ============================================================
# 06_models_tree.R
# CART y Random Forest con tidymodels
# ============================================================
# Team 02 — Problem Set 03
#
# Modelos:
#   CART   — árbol de decisión, tune sobre cost_complexity,
#             tree_depth y min_n
#   RF_1   — Random Forest spec base (misma que OLS_3)
#   RF_2   — Random Forest spec completa + interacciones
#             tune sobre mtry, trees y min_n
#
# CV: espacial (folds_spatial) como métrica principal
# Métrica: MAE en pesos COP
# ============================================================

p_load(tidymodels, rpart, ranger)

# --- Cargar ------------------------------------------------------------------

train_sf      <- readRDS(here(paths$processed, "train_sf.rds"))
test          <- readRDS(here(paths$processed, "test_model.rds"))
folds_spatial <- readRDS(here(paths$processed, "folds_spatial.rds"))

# Recrear folds con train_sf actual
set.seed(SEED)
folds_spatial <- group_vfold_cv(train_sf, group = LocCodigo)

# --- Métrica -----------------------------------------------------------------

metricas <- metric_set(mae, rmse, rsq)

# --- Recipe ------------------------------------------------------------------
# Para árboles: no necesitamos dummies ni normalización
# Los árboles manejan factores directamente
# Predecimos price (no log) — los árboles no lo necesitan

recipe_tree <- recipe(
  price ~
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
  step_dummy(all_nominal_predictors()) |>   # rpart requiere dummies
  step_zv(all_predictors())

# =============================================================
# CART
# =============================================================
# cost_complexity (cp): penalización por complejidad del árbol
#   → más alto = árbol más simple (menos sobreajuste)
# tree_depth: profundidad máxima del árbol
# min_n: mínimo de obs para hacer un split

message("Entrenando CART...")
tic("CART")

spec_cart <- decision_tree(
  cost_complexity = tune(),
  tree_depth      = tune(),
  min_n           = tune()
) |>
  set_engine("rpart") |>
  set_mode("regression")

cart_grid <- grid_regular(
  cost_complexity(range = c(-4, -1)),
  tree_depth(range      = c(3, 10)),
  min_n(range           = c(10, 50)),
  levels = 3   # 3×3×3 = 27 combinaciones
)

wf_cart <- workflow() |>
  add_recipe(recipe_tree) |>
  add_model(spec_cart)

cv_cart <- tune_grid(
  wf_cart,
  resamples = folds_spatial,
  grid      = cart_grid,
  metrics   = metricas,
  control   = control_grid(save_pred = TRUE, verbose = TRUE)
)

toc(log = TRUE)

best_cart <- select_best(cv_cart, metric = "mae")
message("  MAE CV espacial CART: ",
        round(show_best(cv_cart, metric = "mae", n = 1)$mean, 0))
message("  Mejor cp:         ", round(best_cart$cost_complexity, 6))
message("  Mejor tree_depth: ", best_cart$tree_depth)
message("  Mejor min_n:      ", best_cart$min_n)

# =============================================================
# RANDOM FOREST
# =============================================================
# mtry:  número de predictores seleccionados al azar por split
#        → más bajo = más diversidad entre árboles
# trees: número de árboles — más es mejor pero más lento
# min_n: mínimo de obs en nodo terminal
#        → más alto = árboles más simples, menos overfitting

message("Entrenando Random Forest...")
tic("RF")

spec_rf <- rand_forest(
  mtry  = tune(),
  trees = tune(),
  min_n = tune()
) |>
  set_engine("ranger", importance = "impurity") |>
  set_mode("regression")

rf_grid <- grid_regular(
  mtry(range  = c(4, 15)),
  trees(range = c(500, 1000)),
  min_n(range = c(10, 50)),
  levels = 3   # 3×3×3 = 27 combinaciones
)

wf_rf <- workflow() |>
  add_recipe(recipe_tree) |>
  add_model(spec_rf)

cv_rf <- tune_grid(
  wf_rf,
  resamples = folds_spatial,
  grid      = rf_grid,
  metrics   = metricas,
  control   = control_grid(save_pred = TRUE)
)

toc(log = TRUE)

best_rf <- select_best(cv_rf, metric = "mae")
message("  MAE CV espacial RF: ",
        round(show_best(cv_rf, metric = "mae", n = 1)$mean, 0))
message("  Mejor mtry:  ", best_rf$mtry)
message("  Mejor trees: ", best_rf$trees)
message("  Mejor min_n: ", best_rf$min_n)

# =============================================================
# RESUMEN
# =============================================================

resumen_trees <- tibble(
  Modelo = c("CART", "Random Forest"),
  MAE_CV_espacial = c(
    show_best(cv_cart, metric = "mae", n = 1)$mean,
    show_best(cv_rf,   metric = "mae", n = 1)$mean
  )
) |> arrange(MAE_CV_espacial)

message("\n--- Resumen modelos de árboles ---")
print(resumen_trees)

# =============================================================
# FITS FINALES
# =============================================================

message("Entrenando modelos finales sobre todos los datos...")

wf_cart_final <- finalize_workflow(wf_cart, best_cart)
fit_cart      <- fit(wf_cart_final, data = train_sf)

wf_rf_final   <- finalize_workflow(wf_rf, best_rf)
fit_rf        <- fit(wf_rf_final, data = train_sf)

# =============================================================
# SUBMISSIONS
# =============================================================

generar_submission(fit_cart, "CART_cart")
generar_submission(fit_rf,   "RF_rf1")

# =============================================================
# GUARDAR
# =============================================================

saveRDS(fit_cart,      here(paths$training, "fit_cart.rds"))
saveRDS(fit_rf,        here(paths$training, "fit_rf.rds"))
saveRDS(cv_cart,       here(paths$training, "cv_cart.rds"))
saveRDS(cv_rf,         here(paths$training, "cv_rf.rds"))
saveRDS(resumen_trees, here(paths$training, "resumen_trees.rds"))

message("06_models_tree.R ✓")