# ============================================================
# 09_models_sl.R
# Super Learner: tres variantes de ensemble
#   SL v1 — XGBoost fijo + ENET (2 learners, baseline)
#   SL v2 conservador — XGB conservador + XGB agresivo + ENET
#   SL v2 agresivo    — ídem con pesos gaussianos 6km
# ============================================================
# Arquitectura v1 (baseline):
#   SL.xgboost_mod  — XGBoost con parámetros fijos moderados
#   SL.lm_enet      — Elastic Net (α=0.5)
#   Meta-learner: NNLS
#
# Arquitectura v2:
#   SL.xgb_cons  — XGBoost alta regularización, árboles poco profundos
#                  Objetivo: generalizar bien en todo Bogotá
#   SL.xgb_agr   — XGBoost baja regularización, árboles profundos
#                  Objetivo: capturar patrones de precios altos del norte
#   SL.lm_enet   — Elastic Net compartido con v1 (wrapper idéntico)
#   Meta-learner: NNLS
#
# Variantes de pesos:
#   uniform       — toda Bogotá con igual peso
#   gaussianos 6km — más peso a observaciones del sector norte
#
# Variable dependiente: price en COP
# Evaluación: MAE_norte (proxy Kaggle confirmado)
# ============================================================

p_load(SuperLearner, xgboost, glmnet, nnls, dplyr, sf,
       tibble, readr, here, tictoc)

# --- Cargar ------------------------------------------------------------------

train_sf      <- readRDS(here(paths$processed, "train_sf.rds"))
test          <- readRDS(here(paths$processed, "test_model.rds"))
pesos_6km     <- readRDS(here(paths$processed, "pesos_dist_6km.rds"))
pesos_uniform <- readRDS(here(paths$processed, "pesos_uniform.rds"))

# =============================================================
# PARÁMETROS XGB
# =============================================================

# --- XGB fijo (v1) ---------------------------------------------------
# Parámetros moderados, sin búsqueda de hiperparámetros

params_mod <- list(
  objective        = "reg:squarederror",
  eval_metric      = "mae",
  eta              = 0.05,
  max_depth        = 6,
  min_child_weight = 10,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  gamma            = 0.01,
  lambda           = 1,
  alpha            = 0.1,
  nthread          = max(1L, parallel::detectCores() - 1L)
)
nrounds_mod <- 1000

# --- XGB Conservador (v2) --------------------------------------------
# Alta regularización + árboles poco profundos
# Captura la estructura general de precios en Bogotá

params_cons <- list(
  objective        = "reg:absoluteerror",
  eval_metric      = "mae",
  eta              = 0.02,
  max_depth        = 4,
  min_child_weight = 25,
  subsample        = 0.7,
  colsample_bytree = 0.6,
  gamma            = 1.5,
  lambda           = 4,
  alpha            = 1.5,
  nthread          = 7
)
nrounds_cons <- 2000

# --- XGB Agresivo (v2) -----------------------------------------------
# Menor regularización + árboles profundos
# Captura patrones complejos de precios altos del norte

params_agr <- list(
  objective        = "reg:absoluteerror",
  eval_metric      = "mae",
  eta              = 0.05,
  max_depth        = 7,
  min_child_weight = 5,
  subsample        = 0.85,
  colsample_bytree = 0.85,
  gamma            = 0,
  lambda           = 0.5,
  alpha            = 0.05,
  nthread          = 7
)
nrounds_agr <- 1100

message("\nParams XGB fijo (v1):       eta=", params_mod$eta,
        " depth=", params_mod$max_depth,
        " | nrounds=", nrounds_mod)
message("Params XGB conservador (v2): eta=", params_cons$eta,
        " depth=", params_cons$max_depth,
        " mcw=", params_cons$min_child_weight,
        " lambda=", params_cons$lambda,
        " | nrounds=", nrounds_cons)
message("Params XGB agresivo (v2):    eta=", params_agr$eta,
        " depth=", params_agr$max_depth,
        " mcw=", params_agr$min_child_weight,
        " lambda=", params_agr$lambda,
        " | nrounds=", nrounds_agr)

# =============================================================
# FEATURE SPACE
# =============================================================

prep_X_sl <- function(df) {

  vars_usar <- c(
    "property_type", "ESTRATO", "CODIGO_UPZ",
    "surface_total", "surface_covered",
    "rooms", "bedrooms", "bathrooms",
    "EPT", "AREA_HECTA",
    "surface_total_was_na", "surface_covered_was_na",
    "rooms_was_na", "bathrooms_was_na",
    "ESTRATO_was_na", "CODIGO_UPZ_was_na",
    "parqueadero", "parqueadero_inv",
    "piscina", "piscina_privada",
    "ascensor", "sin_ascensor",
    "terraza", "zona_exterior_priv",
    "salon_comunal", "sala_reuniones",
    "porteria", "vigilancia", "conjunto",
    "remodelada", "antiguo", "moderno",
    "cocina_integral", "cocina_americana",
    "gym", "chimenea", "pisos_madera",
    "walk_in_closet", "lavanderia", "duplex",
    "apartaestudio", "vista", "vista_cerros",
    "tenis", "golf", "squash",
    "estudio_cuarto", "escaleras_int",
    "dist_cafe", "dist_bus", "dist_metro",
    "dist_hospital", "dist_colegio", "dist_universidad",
    "dist_parque", "dist_super",
    "n_cafes_500m", "n_rest_500m", "n_farmacias_500m",
    "n_bancos_500m", "n_gym_500m", "n_lamparas_200m",
    "is_residential"
  )

  vars_usar <- intersect(vars_usar, names(df))

  df_sel <- df |>
    st_drop_geometry() |>
    select(all_of(vars_usar)) |>
    mutate(across(where(is.factor), as.character))

  X <- model.matrix(~ . - 1, data = df_sel) |> as.data.frame()
  names(X) <- make.names(names(X), unique = TRUE)
  X
}

if (file.exists(here(paths$training, "X_train_sl.rds"))) {
  X_train <- readRDS(here(paths$training, "X_train_sl.rds"))
  X_test  <- readRDS(here(paths$training, "X_test_sl.rds"))
  message("X_train/X_test cargados desde disco")
} else {
  X_train <- prep_X_sl(train_sf)
  X_test  <- prep_X_sl(test)
  faltantes <- setdiff(names(X_train), names(X_test))
  for (cc in faltantes) X_test[[cc]] <- 0
  X_test <- X_test[, names(X_train), drop = FALSE]
}

y_train <- train_sf$price

message("X_train: ", nrow(X_train), " x ", ncol(X_train))
message("y_train: mean=$", format(round(mean(y_train), 0), big.mark = ","))

# Índices norte / sur
localidades_norte <- c("USAQUEN", "TEUSAQUILLO", "BARRIOS UNIDOS")
idx_norte_train   <- which(train_sf$LocNombre %in% localidades_norte)
idx_sur_train     <- which(!train_sf$LocNombre %in% localidades_norte)

message("  Norte (validation proxy): ", length(idx_norte_train), " obs")
message("  Sur+Chapinero (training):  ", length(idx_sur_train),  " obs")

# =============================================================
# WRAPPERS BASE LEARNERS
# =============================================================

# --- XGB fijo (v1) ---

SL.xgboost_mod <- function(Y, X, newX, family, obsWeights, ...) {
  dtrain  <- xgb.DMatrix(data = as.matrix(X), label = Y, weight = obsWeights)
  dnew    <- xgb.DMatrix(data = as.matrix(newX))
  fit     <- xgb.train(params = params_mod, data = dtrain,
                       nrounds = nrounds_mod, verbose = 0)
  pred    <- predict(fit, dnew)
  fit_obj <- list(object = fit)
  class(fit_obj) <- "SL.xgboost_mod"
  list(pred = pred, fit = fit_obj)
}

predict.SL.xgboost_mod <- function(object, newdata, ...) {
  predict(object$object, xgb.DMatrix(data = as.matrix(newdata)))
}

# --- XGB Conservador (v2) ---

SL.xgb_cons <- function(Y, X, newX, family, obsWeights, ...) {
  dtrain  <- xgb.DMatrix(data = as.matrix(X), label = Y, weight = obsWeights)
  dnew    <- xgb.DMatrix(data = as.matrix(newX))
  fit     <- xgb.train(params = params_cons, data = dtrain,
                       nrounds = nrounds_cons, verbose = 0)
  pred    <- predict(fit, dnew)
  fit_obj <- list(object = fit)
  class(fit_obj) <- "SL.xgb_cons"
  list(pred = pred, fit = fit_obj)
}

predict.SL.xgb_cons <- function(object, newdata, ...) {
  predict(object$object, xgb.DMatrix(data = as.matrix(newdata)))
}

# --- XGB Agresivo (v2) ---

SL.xgb_agr <- function(Y, X, newX, family, obsWeights, ...) {
  dtrain  <- xgb.DMatrix(data = as.matrix(X), label = Y, weight = obsWeights)
  dnew    <- xgb.DMatrix(data = as.matrix(newX))
  fit     <- xgb.train(params = params_agr, data = dtrain,
                       nrounds = nrounds_agr, verbose = 0)
  pred    <- predict(fit, dnew)
  fit_obj <- list(object = fit)
  class(fit_obj) <- "SL.xgb_agr"
  list(pred = pred, fit = fit_obj)
}

predict.SL.xgb_agr <- function(object, newdata, ...) {
  predict(object$object, xgb.DMatrix(data = as.matrix(newdata)))
}

# --- Elastic Net (compartido por v1 y v2) ---

SL.lm_enet <- function(Y, X, newX, family, obsWeights, ...) {
  fit     <- cv.glmnet(x = as.matrix(X), y = Y, alpha = 0.5,
                       nfolds = 5, weights = obsWeights)
  pred    <- as.numeric(predict(fit, newx = as.matrix(newX), s = "lambda.min"))
  fit_obj <- list(object = fit)
  class(fit_obj) <- "SL.lm_enet"
  list(pred = pred, fit = fit_obj)
}

predict.SL.lm_enet <- function(object, newdata, ...) {
  as.numeric(predict(object$object, newx = as.matrix(newdata), s = "lambda.min"))
}

# Librerías por versión
sl.lib_v1 <- c("SL.xgboost_mod", "SL.lm_enet")
sl.lib_v2 <- c("SL.xgb_cons", "SL.xgb_agr", "SL.lm_enet")

# =============================================================
# FOLDS — estratificados por localidad (compartidos)
# =============================================================

set.seed(SEED)
loc_codes <- train_sf |> st_drop_geometry() |> pull(LocCodigo)

folds_index_sl <- split(
  sample(seq_len(nrow(X_train))),
  cut(
    rank(as.numeric(as.factor(loc_codes)), ties.method = "random"),
    breaks = CV_FOLDS,
    labels = FALSE
  )
)

# =============================================================
# SL v1 — pesos uniformes
# =============================================================

message("\n--- SL v1 uniform (XGB fijo + ENET, pesos uniformes) ---")
tic("SL_v1_uniform")

set.seed(SEED)
fit_sl_uniform <- SuperLearner(
  Y          = y_train,
  X          = X_train,
  method     = "method.NNLS",
  SL.library = sl.lib_v1,
  obsWeights = pesos_uniform,
  cvControl  = list(V = CV_FOLDS, validRows = folds_index_sl),
  verbose    = FALSE
)

toc(log = TRUE)
message("Pesos NNLS v1 uniform: ",
        paste(round(fit_sl_uniform$coef, 3), collapse = " | "))

# =============================================================
# SL v1 — pesos gaussianos 6km
# =============================================================

message("\n--- SL v1 6km (XGB fijo + ENET, pesos gaussianos 6km) ---")
tic("SL_v1_6km")

set.seed(SEED)
fit_sl_6km <- SuperLearner(
  Y          = y_train,
  X          = X_train,
  method     = "method.NNLS",
  SL.library = sl.lib_v1,
  obsWeights = pesos_6km,
  cvControl  = list(V = CV_FOLDS, validRows = folds_index_sl),
  verbose    = FALSE
)

toc(log = TRUE)
message("Pesos NNLS v1 6km: ",
        paste(round(fit_sl_6km$coef, 3), collapse = " | "))

# =============================================================
# SL v2 conservador — pesos uniformes
# =============================================================

message("\n--- SL v2 conservador (3 learners, pesos uniformes) ---")
tic("SL_v2_conservador")

set.seed(SEED)
fit_sl_cons <- SuperLearner(
  Y          = y_train,
  X          = X_train,
  method     = "method.NNLS",
  SL.library = sl.lib_v2,
  obsWeights = pesos_uniform,
  cvControl  = list(V = CV_FOLDS, validRows = folds_index_sl),
  verbose    = FALSE
)

toc(log = TRUE)
message("Pesos NNLS v2 conservador:")
message("  XGB_cons=", round(fit_sl_cons$coef["SL.xgb_cons_All"], 3),
        " | XGB_agr=", round(fit_sl_cons$coef["SL.xgb_agr_All"], 3),
        " | ENET=",    round(fit_sl_cons$coef["SL.lm_enet_All"],  3))

# =============================================================
# SL v2 agresivo — pesos gaussianos 6km
# =============================================================

message("\n--- SL v2 agresivo (3 learners, pesos gaussianos 6km) ---")
tic("SL_v2_agresivo")

set.seed(SEED)
fit_sl_agr <- SuperLearner(
  Y          = y_train,
  X          = X_train,
  method     = "method.NNLS",
  SL.library = sl.lib_v2,
  obsWeights = pesos_6km,
  cvControl  = list(V = CV_FOLDS, validRows = folds_index_sl),
  verbose    = FALSE
)

toc(log = TRUE)
message("Pesos NNLS v2 agresivo:")
message("  XGB_cons=", round(fit_sl_agr$coef["SL.xgb_cons_All"], 3),
        " | XGB_agr=", round(fit_sl_agr$coef["SL.xgb_agr_All"], 3),
        " | ENET=",    round(fit_sl_agr$coef["SL.lm_enet_All"],  3))

# =============================================================
# EVALUACIÓN MAE_NORTE — proxy Kaggle
# =============================================================

message("\n--- Evaluación MAE_norte ---")
tic("eval_norte")

eval_mae_norte <- function(lib, pesos_sur) {
  set.seed(SEED)
  fit_temp <- SuperLearner(
    Y          = y_train[idx_sur_train],
    X          = X_train[idx_sur_train, ],
    method     = "method.NNLS",
    SL.library = lib,
    obsWeights = pesos_sur,
    cvControl  = list(
      V = min(CV_FOLDS, max(2, floor(length(idx_sur_train) / 50)))
    ),
    verbose = FALSE
  )
  pred <- as.numeric(
    predict(fit_temp,
            newdata = X_train[idx_norte_train, ],
            onlySL  = TRUE)$pred
  )
  mean(abs(pred - y_train[idx_norte_train]))
}

mae_norte_v1_uniform <- eval_mae_norte(sl.lib_v1, pesos_uniform[idx_sur_train])
mae_norte_v1_6km     <- eval_mae_norte(sl.lib_v1, pesos_6km[idx_sur_train])
mae_norte_cons       <- eval_mae_norte(sl.lib_v2, pesos_uniform[idx_sur_train])
mae_norte_agr        <- eval_mae_norte(sl.lib_v2, pesos_6km[idx_sur_train])

toc(log = TRUE)

# =============================================================
# SUBMISSIONS
# =============================================================

generar_pred_sl <- function(fit_obj) {
  pmax(as.numeric(predict(fit_obj, newdata = X_test, onlySL = TRUE)$pred), 0)
}

pred_v1_uniform <- generar_pred_sl(fit_sl_uniform)
pred_v1_6km     <- generar_pred_sl(fit_sl_6km)
pred_cons       <- generar_pred_sl(fit_sl_cons)
pred_agr        <- generar_pred_sl(fit_sl_agr)

sub_v1_uniform <- tibble(property_id = test$property_id, price = pred_v1_uniform)
sub_v1_6km     <- tibble(property_id = test$property_id, price = pred_v1_6km)
sub_cons       <- tibble(property_id = test$property_id, price = pred_cons)
sub_agr        <- tibble(property_id = test$property_id, price = pred_agr)

for (s in list(sub_v1_uniform, sub_v1_6km, sub_cons, sub_agr)) {
  stopifnot(nrow(s) == nrow(test), !any(is.na(s$price)))
}

write_csv(sub_v1_uniform, here(paths$Super, "Super_sl_v1_uniform.csv"))
write_csv(sub_v1_6km,     here(paths$Super, "Super_sl_v1_6km.csv"))
write_csv(sub_cons,       here(paths$Super, "Super_sl_conservador.csv"))
write_csv(sub_agr,        here(paths$Super, "Super_sl_agresivo.csv"))

message("Submissions generadas:")
message("  Super_sl_v1_uniform.csv")
message("  Super_sl_v1_6km.csv")
message("  Super_sl_conservador.csv")
message("  Super_sl_agresivo.csv")

# =============================================================
# GUARDAR MODELOS
# =============================================================

saveRDS(fit_sl_uniform, here(paths$training, "fit_sl_v1_uniform.rds"))
saveRDS(fit_sl_6km,     here(paths$training, "fit_sl_v1_6km.rds"))
saveRDS(fit_sl_cons,    here(paths$training, "fit_sl_conservador.rds"))
saveRDS(fit_sl_agr,     here(paths$training, "fit_sl_agresivo.rds"))
saveRDS(X_train,        here(paths$training, "X_train_sl.rds"))
saveRDS(X_test,         here(paths$training, "X_test_sl.rds"))

# =============================================================
# RESUMEN FINAL
# =============================================================

resumen_sl <- tibble(
  Modelo        = c("SL v1 uniform", "SL v1 6km",
                    "SL v2 conservador", "SL v2 agresivo"),
  Learners      = c("XGB_mod + ENET", "XGB_mod + ENET",
                    "XGB_cons + XGB_agr + ENET", "XGB_cons + XGB_agr + ENET"),
  Pesos         = c("uniform", "gaussianos 6km", "uniform", "gaussianos 6km"),
  MAE_norte_COP = c(mae_norte_v1_uniform, mae_norte_v1_6km,
                    mae_norte_cons, mae_norte_agr)
) |> arrange(MAE_norte_COP)

saveRDS(resumen_sl, here(paths$training, "resumen_sl.rds"))

message("\n============================================================")
message("  RESUMEN FINAL — SuperLearner (MAE_norte = proxy Kaggle)")
message("============================================================")
print(resumen_sl |> mutate(MAE_norte_COP = format(round(MAE_norte_COP, 0), big.mark = ",")))
message("============================================================\n")

message("09_models_sl.R — listo")
