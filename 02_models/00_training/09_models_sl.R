# =============================================================
# 09_models_superlearner.R
# Super Learner: XGBoost + Elastic Net con pesos por distancia
# =============================================================

p_load(SuperLearner, xgboost, glmnet, nnls, dplyr, sf, tibble, readr, here)

# --- Cargar ------------------------------------------------------------------

train_sf <- readRDS(here(paths$processed, "train_sf.rds"))
test     <- readRDS(here(paths$processed, "test_model.rds"))

# Pesos por distancia a Chapinero (calculados en 04_cv_setup.R)
pesos_6km     <- readRDS(here(paths$processed, "pesos_dist_6km.rds"))
pesos_3km     <- readRDS(here(paths$processed, "pesos_dist_3km.rds"))
pesos_uniform <- readRDS(here(paths$processed, "pesos_uniform.rds"))

# --- Preparar matrices X e y -------------------------------------------------

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

X_train <- prep_X_sl(train_sf)
X_test  <- prep_X_sl(test)

# Alinear columnas train/test
cols_faltantes <- setdiff(names(X_train), names(X_test))
if (length(cols_faltantes) > 0) {
  for (cc in cols_faltantes) X_test[[cc]] <- 0
}
X_test <- X_test[, names(X_train), drop = FALSE]

y_train <- train_sf$price

message("X_train: ", nrow(X_train), " x ", ncol(X_train))
message("y_train: mean=$", format(round(mean(y_train), 0), big.mark = ","))

# Índices para evaluación en cluster norte
localidades_norte <- c("CHAPINERO", "USAQUEN", "TEUSAQUILLO", "BARRIOS UNIDOS")
idx_norte <- which(train_sf$LocNombre %in% localidades_norte)
idx_sur   <- which(!train_sf$LocNombre %in% localidades_norte)

# =============================================================
# WRAPPERS BASE LEARNERS
# obsWeights es donde entran los pesos por distancia
# =============================================================

SL.xgboost_mod <- function(Y, X, newX, family, obsWeights, ...) {
  
  dtrain <- xgb.DMatrix(
    data   = as.matrix(X),
    label  = Y,
    weight = obsWeights
  )
  
  dnew <- xgb.DMatrix(
    data = as.matrix(newX)
  )
  
  params <- list(
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
  
  fit <- xgb.train(
    params   = params,
    data     = dtrain,
    nrounds  = 1000,
    verbose  = 0
  )
  
  pred <- predict(fit, dnew)
  
  # IMPORTANT FIX
  fit_obj <- list(object = fit)
  class(fit_obj) <- "SL.xgboost_mod"
  
  list(
    pred = pred,
    fit  = fit_obj
  )
}

SL.lm_enet <- function(Y, X, newX, family, obsWeights, ...) {
  
  fit <- cv.glmnet(
    x       = as.matrix(X),
    y       = Y,
    alpha   = 0.5,
    nfolds  = 5,
    weights = obsWeights
  )
  
  pred <- predict(
    fit,
    newx = as.matrix(newX),
    s    = "lambda.min"
  )
  
  pred <- as.numeric(pred[, 1])
  
  # IMPORTANT FIX
  fit_obj <- list(object = fit)
  class(fit_obj) <- "SL.lm_enet"
  
  list(
    pred = pred,
    fit  = fit_obj
  )
}

# =============================================================
# PREDICT METHODS (REQUIRED FOR SuperLearner::predict)
# =============================================================

predict.SL.xgboost_mod <- function(object, newdata, ...) {
  
  dnew <- xgb.DMatrix(
    data = as.matrix(newdata)
  )
  
  pred <- predict(
    object$object,
    dnew
  )
  
  pred
}

predict.SL.lm_enet <- function(object, newdata, ...) {
  
  pred <- predict(
    object$object,
    newx = as.matrix(newdata),
    s    = "lambda.min"
  )
  
  as.numeric(pred[, 1])
}

sl.lib <- c(
  "SL.xgboost_mod",
  "SL.lm_enet"
)

# =============================================================
# FOLDS PARA SUPERLEARNER — estratificados por localidad
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
# SUPERLEARNER 1 — pesos uniformes
# =============================================================

message("\n--- SL con pesos uniformes (baseline) ---")
tic("SL_uniform")

set.seed(SEED)
fit_sl_uniform <- SuperLearner(
  Y          = y_train,
  X          = X_train,
  method     = "method.NNLS",
  SL.library = sl.lib,
  obsWeights = pesos_uniform,
  cvControl  = list(V = CV_FOLDS, validRows = folds_index_sl),
  verbose    = FALSE
)

toc(log = TRUE)
message("Pesos NNLS (uniform): ", paste(round(fit_sl_uniform$coef, 3), collapse = " | "))

# =============================================================
# SUPERLEARNER 2 — pesos gaussianos 6km
# =============================================================

message("\n--- SL con pesos gaussianos 6km (foco cluster norte) ---")
tic("SL_pesos_6km")

set.seed(SEED)
fit_sl_6km <- SuperLearner(
  Y          = y_train,
  X          = X_train,
  method     = "method.NNLS",
  SL.library = sl.lib,
  obsWeights = pesos_6km,
  cvControl  = list(V = CV_FOLDS, validRows = folds_index_sl),
  verbose    = FALSE
)

toc(log = TRUE)
message("Pesos NNLS (6km): ", paste(round(fit_sl_6km$coef, 3), collapse = " | "))

# =============================================================
# CV.SuperLearner — estimación honesta
# =============================================================

message("\n--- CV.SuperLearner (estimación honesta — pesos uniform) ---")
tic("CV_SL")

set.seed(SEED)
cv_sl <- CV.SuperLearner(
  Y          = y_train,
  X          = X_train,
  SL.library = sl.lib,
  method     = "method.NNLS",
  obsWeights = pesos_uniform,
  cvControl  = list(V = CV_FOLDS, validRows = folds_index_sl),
  verbose    = FALSE
)

toc(log = TRUE)
print(summary(cv_sl))

# =============================================================
# SAFE EXTRACTION OF BASE LEARNER PREDICTIONS
# =============================================================

xgb_col  <- grep("xgboost", colnames(cv_sl$library.predict), value = TRUE)
enet_col <- grep("enet", colnames(cv_sl$library.predict), value = TRUE)

mae_sl_global <- mean(abs(cv_sl$SL.predict - y_train))
mae_xgb_global <- mean(abs(cv_sl$library.predict[, xgb_col] - y_train))
mae_en_global   <- mean(abs(cv_sl$library.predict[, enet_col] - y_train))

# =============================================================
# EVALUACIÓN EN CLUSTER NORTE — proxy Kaggle
# =============================================================

message("\n--- Evaluando SL en cluster norte (proxy Kaggle) ---")
tic("SL_eval_norte")

set.seed(SEED)
fit_sl_norte_uniform <- SuperLearner(
  Y          = y_train[idx_sur],
  X          = X_train[idx_sur, ],
  method     = "method.NNLS",
  SL.library = sl.lib,
  obsWeights = pesos_uniform[idx_sur],
  cvControl  = list(V = min(CV_FOLDS, max(2, floor(length(idx_sur) / 50)))),
  verbose    = FALSE
)

pred_norte_uniform <- predict(
  fit_sl_norte_uniform,
  newdata = X_train[idx_norte, ],
  onlySL  = TRUE
)$pred
pred_norte_uniform <- as.numeric(pred_norte_uniform)

mae_norte_uniform <- mean(abs(pred_norte_uniform - y_train[idx_norte]))

set.seed(SEED)
fit_sl_norte_6km <- SuperLearner(
  Y          = y_train[idx_sur],
  X          = X_train[idx_sur, ],
  method     = "method.NNLS",
  SL.library = sl.lib,
  obsWeights = pesos_6km[idx_sur],
  cvControl  = list(V = min(CV_FOLDS, max(2, floor(length(idx_sur) / 50)))),
  verbose    = FALSE
)

pred_norte_6km <- predict(
  fit_sl_norte_6km,
  newdata = X_train[idx_norte, ],
  onlySL  = TRUE
)$pred
pred_norte_6km <- as.numeric(pred_norte_6km)

mae_norte_6km <- mean(abs(pred_norte_6km - y_train[idx_norte]))

toc(log = TRUE)

message("  MAE cluster norte — pesos uniform: $",
        format(round(mae_norte_uniform, 0), big.mark = ","))
message("  MAE cluster norte — pesos 6km:    $",
        format(round(mae_norte_6km, 0), big.mark = ","))
message("  → El modelo con menor MAE en cluster norte es el que se submite")

mejor_pesos <- if (mae_norte_6km < mae_norte_uniform) "6km" else "uniform"
fit_sl_final <- if (mejor_pesos == "6km") fit_sl_6km else fit_sl_uniform
message("  → Modelo seleccionado: pesos ", mejor_pesos)

# =============================================================
# RESUMEN FINAL
# =============================================================

resumen_sl <- tibble(
  Modelo  = c(
    "XGBoost base",
    "Elastic Net base",
    "SuperLearner (uniform)",
    "SuperLearner (6km)"
  ),
  CV_tipo = c(
    paste0("global k=", CV_FOLDS),
    paste0("global k=", CV_FOLDS),
    paste0("global k=", CV_FOLDS),
    "cluster norte (proxy Kaggle)"
  ),
  OOF_MAE = c(
    mae_xgb_global,
    mae_en_global,
    mae_sl_global,
    mae_norte_6km
  )
) |> arrange(OOF_MAE)

message("\n============================================================")
message("RESUMEN FINAL — SUPER LEARNER (precio COP)")
message("============================================================")
print(
  resumen_sl |>
    mutate(OOF_MAE = format(round(OOF_MAE, 0), big.mark = ",")),
  n = 15
)
message("\nInterpretación:")
message("  CV global     → estimación estable del error promedio en Bogotá")
message("  Cluster norte → proxy del Kaggle (test = Chapinero)")
message("  Pesos 6km     → más peso a obs similares a Chapinero en entrenamiento")
message("============================================================\n")

# =============================================================
# PREDICCIÓN EN TEST Y SUBMISSION
# =============================================================

pred_sl <- predict(
  fit_sl_final,
  newdata = X_test,
  onlySL  = TRUE
)$pred
pred_sl <- as.numeric(pred_sl)
pred_sl <- pmax(pred_sl, 0)

submission_sl <- tibble(
  property_id = test$property_id,
  price       = pred_sl
)

stopifnot(!any(is.na(submission_sl$price)), nrow(submission_sl) == nrow(test))

write_csv(
  submission_sl,
  here(paths$Super, paste0("Super_sl_", mejor_pesos, ".csv"))
)

otro_pesos <- if (mejor_pesos == "6km") "uniform" else "6km"
fit_sl_otro <- if (otro_pesos == "6km") fit_sl_6km else fit_sl_uniform

pred_sl_otro <- predict(
  fit_sl_otro,
  newdata = X_test,
  onlySL  = TRUE
)$pred
pred_sl_otro <- as.numeric(pred_sl_otro)
pred_sl_otro <- pmax(pred_sl_otro, 0)

submission_sl_otro <- tibble(
  property_id = test$property_id,
  price       = pred_sl_otro
)

write_csv(
  submission_sl_otro,
  here(paths$Super, paste0("Super_sl_", otro_pesos, ".csv"))
)

message("Submissions generadas:")
message("  Super_sl_", mejor_pesos, ".csv  (SELECCIONADO — mej
        or MAE cluster norte)")
message("  Super_sl_", otro_pesos,  ".csv  (alternativa para comparar en Kaggle)")

# =============================================================
# GUARDAR
# =============================================================

saveRDS(fit_sl_uniform,       here(paths$training, "fit_sl_uniform.rds"))
saveRDS(fit_sl_6km,           here(paths$training, "fit_sl_6km.rds"))
saveRDS(cv_sl,                here(paths$training, "cv_sl.rds"))
saveRDS(resumen_sl,           here(paths$training, "resumen_sl.rds"))
saveRDS(X_train,              here(paths$training, "X_train_sl.rds"))
saveRDS(X_test,               here(paths$training, "X_test_sl.rds"))
saveRDS(fit_sl_uniform$coef,  here(paths$training, "sl_pesos_uniform.rds"))
saveRDS(fit_sl_6km$coef,      here(paths$training, "sl_pesos_6km.rds"))

message("09_models_sl.R")

print(mae_sl_global)
mae_norte_uniform
mae_norte_6km