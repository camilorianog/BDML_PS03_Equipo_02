# ============================================================
# 07_models_boost.R
# XGBoost: modelos fijos + Bayesian Optimization (MAE_norte)
# ============================================================

p_load(xgboost, mlrMBO, ParamHelpers, smoof, DiceKriging,
       dplyr, sf, readr, tibble, here, tictoc)

# --- Cargar ------------------------------------------------------------------

train_sf <- readRDS(here(paths$processed, "train_sf.rds"))
test     <- readRDS(here(paths$processed, "test_model.rds"))

# ============================================================
# PREP MATRICES
# ============================================================

# --- Con interacciones (XGB1, XGB2) --------------------------

prep_xgb <- function(df) {
  
  vars <- c(
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
  
  vars <- intersect(vars, names(df))
  
  df_sel <- df |>
    st_drop_geometry() |>
    select(all_of(vars)) |>
    mutate(across(where(is.factor), as.character)) |>
    mutate(
      estrato_surface = as.numeric(ESTRATO) * surface_total,
      estrato_rooms   = as.numeric(ESTRATO) * rooms,
      estrato_bath    = as.numeric(ESTRATO) * bathrooms
    )
  
  X <- model.matrix(
    ~ . +
      property_type:ESTRATO +
      property_type:surface_total +
      CODIGO_UPZ:surface_total - 1,
    data = df_sel
  ) |> as.data.frame()
  
  names(X) <- make.names(names(X), unique = TRUE)
  X
}

# --- Sin interacciones (BO — coherente con SL) ---------------

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

# --- Construir matrices --------------------------------------

X_train_int <- prep_xgb(train_sf)
X_test_int  <- prep_xgb(test)

faltantes <- setdiff(names(X_train_int), names(X_test_int))
for (cc in faltantes) X_test_int[[cc]] <- 0
X_test_int <- X_test_int[, names(X_train_int), drop = FALSE]

if (file.exists(here(paths$training, "X_train_sl.rds"))) {
  X_train_sl <- readRDS(here(paths$training, "X_train_sl.rds"))
  X_test_sl  <- readRDS(here(paths$training, "X_test_sl.rds"))
  message("X_train_sl/X_test_sl cargados desde disco")
} else {
  X_train_sl <- prep_X_sl(train_sf)
  X_test_sl  <- prep_X_sl(test)
  faltantes_sl <- setdiff(names(X_train_sl), names(X_test_sl))
  for (cc in faltantes_sl) X_test_sl[[cc]] <- 0
  X_test_sl <- X_test_sl[, names(X_train_sl), drop = FALSE]
}

y_train <- train_sf$price

message("X_train (interacciones): ", nrow(X_train_int), " x ", ncol(X_train_int))
message("X_train (SL/BO):         ", nrow(X_train_sl),  " x ", ncol(X_train_sl))

# --- Índices norte / sur (BO + proxy Kaggle) -----------------

localidades_norte <- c("USAQUEN", "TEUSAQUILLO", "BARRIOS UNIDOS")
idx_norte_train   <- which(train_sf$LocNombre %in% localidades_norte)
idx_sur_train     <- which(!train_sf$LocNombre %in% localidades_norte)

stopifnot(length(intersect(idx_norte_train, idx_sur_train)) == 0)
stopifnot(length(idx_norte_train) + length(idx_sur_train) == nrow(X_train_sl))

message("Norte (validation proxy): ", length(idx_norte_train), " obs")
message("Sur+Chapinero (training):  ", length(idx_sur_train),  " obs")

X_sur   <- X_train_sl[idx_sur_train,   ]
X_norte <- X_train_sl[idx_norte_train, ]
y_sur   <- y_train[idx_sur_train]
y_norte <- y_train[idx_norte_train]

# --- DMatrix (modelos fijos) ---------------------------------

dtrain_int <- xgb.DMatrix(data = as.matrix(X_train_int), label = y_train)
dtest_int  <- xgb.DMatrix(data = as.matrix(X_test_int))

n_threads <- 6

# ============================================================
# XGB1 — conservador
# ============================================================

params_xgb1 <- list(
  objective        = "reg:squarederror",
  eval_metric      = "mae",
  eta              = 0.03,
  max_depth        = 4,
  min_child_weight = 15,
  subsample        = 0.75,
  colsample_bytree = 0.75,
  gamma            = 1,
  lambda           = 3,
  alpha            = 1,
  nthread          = n_threads
)

message("\n--- XGB1 (conservador) ---")
set.seed(SEED)

xgb1 <- xgb.train(
  params  = params_xgb1,
  data    = dtrain_int,
  nrounds = 600,
  evals   = list(train = dtrain_int),
  verbose = 1
)

pred_xgb1 <- pmax(predict(xgb1, dtest_int), 0)

# ============================================================
# XGB2 — extra regularizado
# ============================================================

params_xgb2 <- list(
  objective        = "reg:absoluteerror",
  eval_metric      = "mae",
  eta              = 0.02,
  max_depth        = 3,
  min_child_weight = 30,
  subsample        = 0.6,
  colsample_bytree = 0.6,
  gamma            = 5,
  lambda           = 10,
  alpha            = 5,
  nthread          = n_threads
)

message("\n--- XGB2 (extra regularizado) ---")
set.seed(SEED)

xgb2 <- xgb.train(
  params  = params_xgb2,
  data    = dtrain_int,
  nrounds = 800,
  evals   = list(train = dtrain_int),
  verbose = 1
)

pred_xgb2 <- pmax(predict(xgb2, dtest_int), 0)

# ============================================================
# BAYESIAN OPTIMIZATION — objetivo: MAE_norte
# ============================================================
# Entrena en idx_sur_train, evalúa en idx_norte_train.
# Los params ganadores son directamente aplicables como
# params_agr en 09_models_sl.R.

NROUNDS_BO <- 1000

obj_fun <- smoof::makeSingleObjectiveFunction(
  
  name = "xgb_mae_norte_sl",
  
  fn = function(x) {
    
    x <- as.list(x)
    
    params <- list(
      objective        = "reg:squarederror",
      eval_metric      = "mae",
      eta              = x[["eta"]],
      max_depth        = as.integer(x[["max_depth"]]),
      min_child_weight = x[["min_child_weight"]],
      subsample        = x[["subsample"]],
      colsample_bytree = x[["colsample_bytree"]],
      gamma            = x[["gamma"]],
      lambda           = x[["lambda"]],
      alpha            = x[["alpha"]],
      nthread          = 4
    )
    
    mae_norte <- tryCatch({
      dtrain <- xgb.DMatrix(data = as.matrix(X_sur), label = y_sur)
      fit    <- xgb.train(params = params, data = dtrain,
                          nrounds = NROUNDS_BO, verbose = 0)
      pred   <- predict(fit, xgb.DMatrix(data = as.matrix(X_norte)))
      mean(abs(pred - y_norte))
    }, error = function(e) {
      message("  ERROR: ", conditionMessage(e))
      Inf
    })
    
    message(
      "  MAE_norte=$", format(round(mae_norte, 0), big.mark = ","),
      " | depth=", as.integer(x[["max_depth"]]),
      " eta=", round(x[["eta"]], 3),
      " mcw=", round(x[["min_child_weight"]], 1),
      " lambda=", round(x[["lambda"]], 2),
      " alpha=", round(x[["alpha"]], 2)
    )
    
    return(mae_norte)
  },
  
  par.set = makeParamSet(
    makeNumericParam("eta",              lower = 0.02, upper = 0.08),
    makeIntegerParam("max_depth",        lower = 5,    upper = 7),
    makeNumericParam("min_child_weight", lower = 5,    upper = 20),
    makeNumericParam("subsample",        lower = 0.65, upper = 0.90),
    makeNumericParam("colsample_bytree", lower = 0.65, upper = 0.90),
    makeNumericParam("gamma",            lower = 0,    upper = 2),
    makeNumericParam("lambda",           lower = 0,    upper = 3),
    makeNumericParam("alpha",            lower = 0,    upper = 1)
  ),
  
  minimize = TRUE
)

surrogate <- makeLearner(
  "regr.km",
  predict.type = "se",
  config       = list(show.learner.output = FALSE)
)

ctrl <- makeMBOControl()
ctrl <- setMBOControlTermination(ctrl, iters = 30)
ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())

ps <- ParamHelpers::getParamSet(obj_fun)

set.seed(SEED)
design_lhs <- generateDesign(n = 9, par.set = ps)

design_anchor <- data.frame(
  eta              = 0.05,
  max_depth        = 6,
  min_child_weight = 10,
  subsample        = 0.85,
  colsample_bytree = 0.85,
  gamma            = 0.01,
  lambda           = 0.8,
  alpha            = 0.05
)

design <- rbind(design_lhs, design_anchor)

message("\n--- Bayesian Optimization (MAE_norte) ---")
message("  Train: ", length(idx_sur_train),  " obs")
message("  Valid: ", length(idx_norte_train), " obs")
message("  nrounds fijo: ", NROUNDS_BO)
message("  9 LHS + 1 anchor + 30 BO = 40 evaluaciones")
tic("BO norte")

set.seed(SEED)
res_mbo <- mbo(
  fun       = obj_fun,
  design    = design,
  learner   = surrogate,
  control   = ctrl,
  show.info = TRUE
)

toc(log = TRUE)

message("\nMejor configuración BO:")
print(as.data.frame(res_mbo$x))
message("MAE_norte: $", format(round(res_mbo$y, 0), big.mark = ","))

opt_path <- as.data.frame(res_mbo$opt.path)
message("\nTop 10:")
print(
  opt_path |>
    arrange(y) |>
    head(10) |>
    select(eta, max_depth, min_child_weight,
           subsample, colsample_bytree,
           gamma, lambda, alpha, y)
)

write_csv(
  opt_path |> arrange(y),
  here(paths$Boosting, "bo_norte_opt_path.csv")
)

# --- Refinar nrounds con early stopping ----------------------

best_params <- list(
  objective        = "reg:squarederror",
  eval_metric      = "mae",
  eta              = res_mbo$x[["eta"]],
  max_depth        = as.integer(res_mbo$x[["max_depth"]]),
  min_child_weight = res_mbo$x[["min_child_weight"]],
  subsample        = res_mbo$x[["subsample"]],
  colsample_bytree = res_mbo$x[["colsample_bytree"]],
  gamma            = res_mbo$x[["gamma"]],
  lambda           = res_mbo$x[["lambda"]],
  alpha            = res_mbo$x[["alpha"]],
  nthread          = 7
)

message("\nRefinando nrounds con early stopping en split norte...")
dtrain_sur   <- xgb.DMatrix(data = as.matrix(X_sur),   label = y_sur)
dvalid_norte <- xgb.DMatrix(data = as.matrix(X_norte),  label = y_norte)

set.seed(SEED)
fit_refine <- xgb.train(
  params                = best_params,
  data                  = dtrain_sur,
  nrounds               = 3000,
  early_stopping_rounds = 100,
  evals                 = list(norte = dvalid_norte),
  verbose               = 1
)

best_iter <- fit_refine$best_iteration
if (is.null(best_iter) || length(best_iter) == 0) {
  message("best_iteration NULL -> usando fallback")
  best_iter <- NROUNDS_BO
}
best_nrounds <- round(best_iter * 1.10)

pred_norte_final <- predict(fit_refine, dvalid_norte)
mae_norte_bo     <- mean(abs(pred_norte_final - y_norte))

message("  MAE_norte (params BO + early stop): $",
        format(round(mae_norte_bo, 0), big.mark = ","))

# --- Modelo final en training completo -----------------------

dtrain_full <- xgb.DMatrix(data = as.matrix(X_train_sl), label = y_train)
dtest_sl    <- xgb.DMatrix(data = as.matrix(X_test_sl))

set.seed(SEED)
xgb_bo <- xgb.train(
  params  = best_params,
  data    = dtrain_full,
  nrounds = best_nrounds,
  verbose = 1
)

pred_bo <- pmax(predict(xgb_bo, dtest_sl), 0)

# ============================================================
# SUBMISSIONS
# ============================================================

dir.create(here(paths$Boosting), recursive = TRUE, showWarnings = FALSE)

sub_xgb1 <- tibble(property_id = test$property_id, price = pred_xgb1)
sub_xgb2 <- tibble(property_id = test$property_id, price = pred_xgb2)
sub_bo   <- tibble(property_id = test$property_id, price = pred_bo)

write_csv(sub_xgb1, here(paths$Boosting, "xgb1.csv"))
write_csv(sub_xgb2, here(paths$Boosting, "xgb2.csv"))
write_csv(sub_bo,   here(paths$Boosting, "XGB_bo_norte.csv"))

message("Submissions generadas:")
message("  xgb1.csv")
message("  xgb2.csv")
message("  XGB_bo_norte.csv")

# ============================================================
# GUARDAR MODELOS
# ============================================================

saveRDS(xgb1,   here(paths$Boosting, "xgb1.rds"))
saveRDS(xgb2,   here(paths$Boosting, "xgb2.rds"))
saveRDS(xgb_bo, here(paths$training, "xgb_bo_norte.rds"))

write_csv(opt_path |> arrange(y), here(paths$Boosting, "bo_norte_opt_path.csv"))

# ============================================================
# RESUMEN
# ============================================================

resumen_boost <- tibble(
  Modelo      = c("XGB1 (conservador)", "XGB2 (extra regularizado)", "XGB_BO (norte)"),
  Pred_media  = c(mean(pred_xgb1), mean(pred_xgb2), mean(pred_bo)),
  MAE_norte   = c(NA, NA, mae_norte_bo)
)

message("\n============================================================")
message("  RESUMEN FINAL — XGBoost")
message("============================================================")
print(
  resumen_boost |>
    mutate(
      Pred_media = format(round(Pred_media, 0), big.mark = ","),
      MAE_norte  = ifelse(is.na(MAE_norte), "—",
                          format(round(MAE_norte, 0), big.mark = ","))
    )
)
message("============================================================\n")

message("07_models_boost.R — listo")
