# ============================================================
# 10_analysis.R
# Best Model Deep Dive + Spatial Validation + Business Analysis
# ============================================================

# ============================================================
# CARGAR
# ============================================================

p_load(
  tidyverse,
  sf,
  xgboost,
  yardstick,
  patchwork,
  gt,
  tictoc,
  here,
  purrr,
  scales,
  rsample
)

train_sf <- readRDS(here(paths$processed, "train_sf.rds"))
test     <- readRDS(here(paths$processed, "test_model.rds"))

folds_std         <- readRDS(here(paths$processed, "folds_std.rds"))
folds_global      <- readRDS(here(paths$processed, "folds_global.rds"))
folds_spatial     <- readRDS(here(paths$processed, "folds_spatial.rds"))
folds_spatial_upz <- readRDS(here(paths$processed, "folds_spatial_upz.rds"))
folds_norte       <- readRDS(here(paths$processed, "folds_norte.rds"))

# ============================================================
# HELPER EXPORT GT TABLES
# ============================================================

save_gt_table <- function(df, file_name, digits = 2) {
  
  gt_tbl <- df |>
    
    gt() |>
    
    fmt_number(
      columns = where(is.numeric),
      decimals = digits
    ) |>
    
    tab_options(
      table.font.size = px(12),
      data_row.padding = px(4)
    )
  
  gtsave(
    gt_tbl,
    here(paths$figures, paste0(file_name, ".png"))
  )
  
  gtsave(
    gt_tbl,
    here(paths$tables, paste0(file_name, ".html"))
  )
}

# ============================================================
# PREP
# ============================================================

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
  
  X <- model.matrix(~ . - 1, data = df_sel) |>
    as.data.frame()
  
  names(X) <- make.names(names(X), unique = TRUE)
  
  X
}

X_train <- prep_X_sl(train_sf)
X_test  <- prep_X_sl(test)

faltantes <- setdiff(names(X_train), names(X_test))

for (cc in faltantes) {
  X_test[[cc]] <- 0
}

X_test <- X_test[, names(X_train), drop = FALSE]

y_train <- train_sf$price

# ============================================================
# BEST MODEL
# ============================================================

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

# ============================================================
# METRICS
# ============================================================

calc_metrics <- function(y_true, y_pred) {
  
  err <- y_pred - y_true
  ae  <- abs(err)
  ape <- ae / y_true
  
  tibble(
    MAE                 = mean(ae, na.rm = TRUE),
    Median_AE           = median(ae, na.rm = TRUE),
    RMSE                = sqrt(mean(err^2, na.rm = TRUE)),
    MAPE                = mean(ape, na.rm = TRUE),
    Median_APE          = median(ape, na.rm = TRUE),
    P90_APE             = quantile(ape, 0.90, na.rm = TRUE),
    Bias                = mean(err, na.rm = TRUE),
    Overprediction_Pct  = mean(err > 0, na.rm = TRUE),
    Underprediction_Pct = mean(err < 0, na.rm = TRUE),
    P90_AE              = quantile(ae, 0.90, na.rm = TRUE),
    P95_AE              = quantile(ae, 0.95, na.rm = TRUE)
  )
}

# ============================================================
# CV FUNCTION
# ============================================================

cv_xgb <- function(folds, nombre_cv) {
  
  pred_oof <- rep(NA_real_, nrow(X_train))
  fold_mae <- c()
  
  for (i in seq_along(folds$splits)) {
    
    split_i <- folds$splits[[i]]
    
    train_data <- rsample::analysis(split_i)
    valid_data <- rsample::assessment(split_i)
    
    idx_train <- as.integer(rownames(train_data))
    idx_valid <- as.integer(rownames(valid_data))
    
    idx_train <- sort(unique(idx_train))
    idx_valid <- sort(unique(idx_valid))
    
    message(
      nombre_cv,
      " | Fold ", i,
      " | train=", length(idx_train),
      " | valid=", length(idx_valid)
    )
    
    dtrain <- xgb.DMatrix(
      data  = as.matrix(X_train[idx_train, , drop = FALSE]),
      label = y_train[idx_train]
    )
    
    dvalid <- xgb.DMatrix(
      data = as.matrix(X_train[idx_valid, , drop = FALSE])
    )
    
    fit <- xgb.train(
      params  = params_mod,
      data    = dtrain,
      nrounds = nrounds_mod,
      verbose = 0
    )
    
    pred <- predict(fit, dvalid)
    
    pred_oof[idx_valid] <- pred
    
    mae_fold <- mean(abs(pred - y_train[idx_valid]))
    
    fold_mae <- c(fold_mae, mae_fold)
    
    message(
      nombre_cv,
      " | Fold ",
      i,
      " | n_train=",
      length(idx_train),
      " | n_valid=",
      length(idx_valid),
      " | MAE = ",
      format(round(mae_fold, 0), big.mark = ",")
    )
  }
  
  list(
    pred = pred_oof,
    fold_mae = fold_mae,
    metrics = calc_metrics(y_train, pred_oof)
  )
}

# ============================================================
# RUN CV
# ============================================================

tic("Random CV")
cv_std <- cv_xgb(folds_std, "Random CV")
toc()

tic("Global CV")
cv_global <- cv_xgb(
  folds_global,
  "Random CV (stratified locality)"
)
toc()

tic("Spatial locality")
cv_spatial <- cv_xgb(
  folds_spatial,
  "Spatial CV (leave-locality-out)"
)
toc()

tic("Spatial UPZ")
cv_upz <- cv_xgb(
  folds_spatial_upz,
  "Spatial CV (leave-UPZ-out)"
)
toc()

tic("Northeast holdout")
cv_norte <- cv_xgb(
  folds_norte,
  "Northeast Cluster Holdout"
)
toc()

# ============================================================
# MAIN CV TABLE
# ============================================================

tabla_cv <- bind_rows(
  
  cv_std$metrics |>
    mutate(CV = "Random CV"),
  
  cv_global$metrics |>
    mutate(CV = "Random CV (stratified locality)"),
  
  cv_spatial$metrics |>
    mutate(CV = "Spatial CV (leave-locality-out)"),
  
  cv_upz$metrics |>
    mutate(CV = "Spatial CV (leave-UPZ-out)"),
  
  cv_norte$metrics |>
    mutate(CV = "Northeast Cluster Holdout")
  
) |>
  
  select(CV, everything())

print(tabla_cv)

write_csv(
  tabla_cv,
  here(paths$tables, "tabla_cv_completa.csv")
)

save_gt_table(
  tabla_cv,
  "tabla_cv_completa"
)

# ============================================================
# FOLD TABLE
# ============================================================

tabla_folds <- tibble(
  
  CV = c(
    rep("Random CV", length(cv_std$fold_mae)),
    rep("Random CV (stratified locality)", length(cv_global$fold_mae)),
    rep("Spatial CV (leave-locality-out)", length(cv_spatial$fold_mae)),
    rep("Spatial CV (leave-UPZ-out)", length(cv_upz$fold_mae)),
    rep("Northeast Cluster Holdout", length(cv_norte$fold_mae))
  ),
  
  Fold = c(
    seq_along(cv_std$fold_mae),
    seq_along(cv_global$fold_mae),
    seq_along(cv_spatial$fold_mae),
    seq_along(cv_upz$fold_mae),
    seq_along(cv_norte$fold_mae)
  ),
  
  MAE = c(
    cv_std$fold_mae,
    cv_global$fold_mae,
    cv_spatial$fold_mae,
    cv_upz$fold_mae,
    cv_norte$fold_mae
  )
)

write_csv(
  tabla_folds,
  here(paths$tables, "tabla_mae_folds.csv")
)

save_gt_table(
  tabla_folds,
  "tabla_mae_folds"
)

# ============================================================
# RANDOM VS SPATIAL GAP
# ============================================================

mae_std     <- tabla_cv$MAE[tabla_cv$CV == "Random CV"]
mae_spatial <- tabla_cv$MAE[tabla_cv$CV == "Spatial CV (leave-locality-out)"]
mae_upz     <- tabla_cv$MAE[tabla_cv$CV == "Spatial CV (leave-UPZ-out)"]
mae_norte   <- tabla_cv$MAE[tabla_cv$CV == "Northeast Cluster Holdout"]

gap_spatial <- 100 * (mae_spatial / mae_std - 1)
gap_upz     <- 100 * (mae_upz / mae_std - 1)
gap_norte   <- 100 * (mae_norte / mae_std - 1)

tabla_gap <- tibble(
  
  Validation = c(
    "Spatial locality",
    "Spatial UPZ",
    "Northeast holdout"
  ),
  
  MAE = c(
    mae_spatial,
    mae_upz,
    mae_norte
  ),
  
  Gap_vs_Random_Pct = c(
    gap_spatial,
    gap_upz,
    gap_norte
  )
)

write_csv(
  tabla_gap,
  here(paths$tables, "tabla_gap_spatial.csv")
)

save_gt_table(
  tabla_gap,
  "tabla_gap_spatial"
)

p_gap <- tabla_gap |>
  
  ggplot(aes(
    x = Validation,
    y = Gap_vs_Random_Pct
  )) +
  
  geom_col() +
  
  labs(
    title = "Spatial Validation Penalty",
    subtitle = "Increase in MAE relative to random CV",
    x = "",
    y = "% increase in MAE"
  )

ggsave(
  here(paths$figures, "spatial_gap.png"),
  p_gap,
  width = 8,
  height = 5
)

# ============================================================
# TRAIN FINAL MODEL
# ============================================================

dtrain_full <- xgb.DMatrix(
  data  = as.matrix(X_train),
  label = y_train
)

fit_final <- xgb.train(
  params  = params_mod,
  data    = dtrain_full,
  nrounds = nrounds_mod,
  verbose = 0
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_tbl <- xgb.importance(
  model = fit_final,
  feature_names = names(X_train)
)

importance_tbl <- importance_tbl |>
  
  mutate(
    
    feature_group = case_when(
      
      str_detect(
        Feature,
        "dist_|n_|AREA_HECTA|EPT"
      ) ~ "OSM / Spatial",
      
      str_detect(
        Feature,
        "vista|duplex|terraza|chimenea|cocina|gym|walk_in|lavanderia|moderno|remodelada"
      ) ~ "Text-derived / amenities",
      
      TRUE ~ "Properati structural"
    )
  )

write_csv(
  importance_tbl,
  here(paths$tables, "xgb_importance_grouped.csv")
)

group_importance <- importance_tbl |>
  
  group_by(feature_group) |>
  
  summarise(
    Total_Gain = sum(Gain, na.rm = TRUE),
    Mean_Gain  = mean(Gain, na.rm = TRUE),
    n_features = n(),
    .groups = "drop"
  ) |>
  
  mutate(
    Gain_Share = Total_Gain / sum(Total_Gain)
  ) |>
  
  arrange(desc(Total_Gain))

write_csv(
  group_importance,
  here(paths$tables, "feature_group_importance.csv")
)

save_gt_table(
  group_importance,
  "feature_group_importance"
)

p_imp <- importance_tbl |>
  slice_max(Gain, n = 20) |>
  ggplot(aes(
    x = reorder(Feature, Gain),
    y = Gain
  )) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Top 20 Variables — XGBoost",
    x = "",
    y = "Gain"
  )

ggsave(
  here(paths$figures, "xgb_gain_importance.png"),
  p_imp,
  width = 10,
  height = 8
)

p_group_imp <- group_importance |>
  
  ggplot(aes(
    x = reorder(feature_group, Gain_Share),
    y = Gain_Share
  )) +
  
  geom_col() +
  
  coord_flip() +
  
  scale_y_continuous(labels = scales::percent) +
  
  labs(
    title = "Predictive Power by Feature Group",
    x = "",
    y = "Share of total XGBoost gain"
  )

ggsave(
  here(paths$figures, "feature_group_importance.png"),
  p_group_imp,
  width = 8,
  height = 5
)

# ============================================================
# SHAP VALUES
# ============================================================

set.seed(SEED)

idx_shap <- sample(
  seq_len(nrow(X_train)),
  min(2000, nrow(X_train))
)

shap <- predict(
  fit_final,
  newdata = xgb.DMatrix(as.matrix(X_train[idx_shap, ])),
  predcontrib = TRUE
)

shap_df <- as.data.frame(shap)

# Eliminar columna bias independientemente del nombre
bias_cols <- names(shap_df)[
  str_detect(
    names(shap_df),
    regex("bias", ignore_case = TRUE)
  )
]

shap_df_clean <- shap_df |>
  select(-all_of(bias_cols))

shap_long <- shap_df_clean |>
  
  pivot_longer(
    cols = everything(),
    names_to = "feature",
    values_to = "shap"
  ) |>
  
  group_by(feature) |>
  
  summarise(
    mean_abs_shap = mean(abs(shap), na.rm = TRUE),
    .groups = "drop"
  ) |>
  
  arrange(desc(mean_abs_shap))

write_csv(
  shap_long,
  here(paths$tables, "shap_summary.csv")
)

save_gt_table(
  shap_long |> slice_head(n = 25),
  "shap_summary_top25"
)

p_shap <- shap_long |>
  
  slice_max(mean_abs_shap, n = 20) |>
  
  ggplot(aes(
    x = reorder(feature, mean_abs_shap),
    y = mean_abs_shap
  )) +
  
  geom_col() +
  
  coord_flip() +
  
  labs(
    title = "Top SHAP Features",
    x = "",
    y = "Mean |SHAP|"
  )

ggsave(
  here(paths$figures, "shap_importance.png"),
  p_shap,
  width = 10,
  height = 8
)

# ============================================================
# SHAP DIRECTIONAL ANALYSIS
# ============================================================

features_validas <- names(shap_df_clean)[
  
  map_lgl(
    names(shap_df_clean),
    ~ .x %in% names(X_train) &&
      is.numeric(X_train[[.x]]) &&
      is.numeric(shap_df_clean[[.x]])
  )
]

shap_corr <- tibble(
  feature = features_validas
) |>
  
  mutate(
    
    correlation = map_dbl(
      feature,
      ~cor(
        as.numeric(X_train[idx_shap, .x]),
        as.numeric(shap_df_clean[[.x]]),
        use = "complete.obs"
      )
    )
  ) |>
  
  arrange(desc(abs(correlation)))

write_csv(
  shap_corr,
  here(paths$tables, "shap_directionality.csv")
)

save_gt_table(
  shap_corr,
  "shap_directionality"
)

# ============================================================
# PREPARAR TABLA DE BIAS
# ============================================================

bias_tbl <- bind_rows(
  
  tibble(
    CV = "Random CV",
    y = y_train,
    pred = cv_std$pred
  ),
  
  tibble(
    CV = "Spatial locality",
    y = y_train,
    pred = cv_spatial$pred
  ),
  
  tibble(
    CV = "Spatial UPZ",
    y = y_train,
    pred = cv_upz$pred
  ),
  
  tibble(
    CV = "Northeast Holdout",
    y = y_train,
    pred = cv_norte$pred
  )
  
) |>
  
  mutate(
    
    error = pred - y,
    
    abs_error = abs(error),
    
    pct_error = abs_error / y,
    
    estrato = rep(train_sf$ESTRATO, 4),
    
    localidad = rep(train_sf$LocNombre, 4),
    
    property_type = rep(train_sf$property_type, 4)
  )

# ============================================================
# BIAS BY ESTRATO
# ============================================================

bias_estrato <- bias_tbl |>
  
  group_by(CV, estrato) |>
  
  summarise(
    MAE = mean(abs_error, na.rm = TRUE),
    MAPE = mean(pct_error, na.rm = TRUE),
    Bias = mean(error, na.rm = TRUE),
    N = n(),
    .groups = "drop"
  )

write_csv(
  bias_estrato,
  here(paths$tables, "bias_by_estrato.csv")
)

save_gt_table(
  bias_estrato,
  "bias_by_estrato"
)

# ============================================================
# BIAS BY LOCALIDAD
# ============================================================

bias_localidad <- bias_tbl |>
  
  group_by(CV, localidad) |>
  
  summarise(
    MAE = mean(abs_error, na.rm = TRUE),
    MAPE = mean(pct_error, na.rm = TRUE),
    Bias = mean(error, na.rm = TRUE),
    N = n(),
    .groups = "drop"
  )

write_csv(
  bias_localidad,
  here(paths$tables, "bias_by_localidad.csv")
)

save_gt_table(
  bias_localidad,
  "bias_by_localidad"
)

# ============================================================
# CATASTROPHIC ERRORS
# ============================================================

top_errors <- bias_tbl |>
  
  arrange(desc(abs_error)) |>
  
  slice_head(n = 100) |>
  
  select(
    CV,
    localidad,
    estrato,
    property_type,
    y,
    pred,
    error,
    abs_error,
    pct_error
  )

write_csv(
  catastrophic_summary,
  here(paths$tables, "catastrophic_error_summary.csv")
)

save_gt_table(
  catastrophic_summary,
  "catastrophic_error_summary"
)

catastrophic_summary <- top_errors |>
  
  group_by(localidad) |>
  
  summarise(
    N_catastrophic = n(),
    Mean_abs_error = mean(abs_error),
    .groups = "drop"
  ) |>
  
  arrange(desc(N_catastrophic))

write_csv(
  catastrophic_summary,
  here(paths$tables, "catastrophic_error_summary.csv")
)

# ============================================================
# RISK ANALYSIS
# ============================================================

risk_tbl <- bias_tbl |>
  
  group_by(CV) |>
  
  summarise(
    Overprediction_Pct = mean(error > 0, na.rm = TRUE),
    Underprediction_Pct = mean(error < 0, na.rm = TRUE),
    Mean_Overprediction = mean(error[error > 0], na.rm = TRUE),
    Mean_Underprediction = mean(error[error < 0], na.rm = TRUE),
    .groups = "drop"
  )

write_csv(
  risk_tbl,
  here(paths$tables, "risk_analysis.csv")
)

# ============================================================
# BENCHMARK
# ============================================================

loc_mean_tbl <- train_sf |>
  
  st_drop_geometry() |>
  
  group_by(LocNombre) |>
  
  summarise(
    loc_mean_price = mean(price, na.rm = TRUE),
    .groups = "drop"
  )

naive_pred <- train_sf |>
  
  st_drop_geometry() |>
  
  left_join(loc_mean_tbl, by = "LocNombre") |>
  
  pull(loc_mean_price)

naive_mae <- mean(abs(naive_pred - y_train))

benchmark_tbl <- tibble(
  
  Model = c(
    "Naive locality mean",
    "XGBoost Random CV",
    "XGBoost Spatial CV",
    "XGBoost UPZ CV",
    "XGBoost Northeast Holdout"
  ),
  
  MAE = c(
    naive_mae,
    cv_std$metrics$MAE,
    cv_spatial$metrics$MAE,
    cv_upz$metrics$MAE,
    cv_norte$metrics$MAE
  )
)

write_csv(
  benchmark_tbl,
  here(paths$tables, "benchmark_models.csv")
)

save_gt_table(
  benchmark_tbl,
  "benchmark_models"
)

# ============================================================
# CALIBRATION
# ============================================================

calib_df <- tibble(
  actual = y_train,
  pred   = cv_spatial$pred
)

p_calib <- ggplot(calib_df, aes(actual, pred)) +
  geom_point(alpha = 0.2) +
  geom_abline(slope = 1, intercept = 0) +
  labs(
    title = "Calibration Plot",
    x = "Actual price",
    y = "Predicted price"
  )

ggsave(
  here(paths$figures, "calibration_plot.png"),
  p_calib,
  width = 8,
  height = 8
)

# ============================================================
# ERROR VS PRICE
# ============================================================

p_price_error <- bias_tbl |>
  
  ggplot(aes(y, abs_error)) +
  
  geom_point(alpha = 0.1) +
  
  geom_smooth(se = FALSE) +
  
  scale_x_log10() +
  
  scale_y_log10() +
  
  facet_wrap(~CV) +
  
  labs(
    title = "Absolute Error vs Price",
    x = "Price",
    y = "Absolute Error"
  )

ggsave(
  here(paths$figures, "error_vs_price.png"),
  p_price_error,
  width = 12,
  height = 8
)

# ============================================================
# PREPARAR MAPAS DE ERROR
# ============================================================

map_spatial <- train_sf |>
  
  mutate(
    
    error = abs(cv_spatial$pred - price)
  ) |>
  
  filter(!is.na(error)) |>
  
  mutate(
    
    error_cap = pmin(
      error,
      quantile(error, 0.95, na.rm = TRUE)
    )
  )

map_upz <- train_sf |>
  
  mutate(
    
    error = abs(cv_upz$pred - price)
  ) |>
  
  filter(!is.na(error)) |>
  
  mutate(
    
    error_cap = pmin(
      error,
      quantile(error, 0.95, na.rm = TRUE)
    )
  )

map_norte <- train_sf |>
  
  mutate(
    
    error = abs(cv_norte$pred - price)
  ) |>
  
  filter(!is.na(error)) |>
  
  mutate(
    
    error_cap = pmin(
      error,
      quantile(error, 0.95, na.rm = TRUE)
    )
  )

# ============================================================
# MAPA BASE RECORTADO
# ============================================================

localidades_map <- st_read(
  here(paths$raw, "Loca.shp"),
  quiet = TRUE
) |>
  
  st_transform(4326) |>
  
  st_make_valid() |>
  
  filter(
    !LocNombre %in% c(
      "SUMAPAZ",
      "USME",
      "CIUDAD BOLIVAR",
      "BOSA"
    )
  )

# bbox basada en observaciones reales
bbox_data <- st_bbox(train_sf)

# ============================================================
# MAPA SPATIAL CV
# ============================================================

p_map_spatial <- ggplot() +
  
  geom_sf(
    data = localidades_map,
    fill = "gray97",
    color = "gray75",
    linewidth = 0.25
  ) +
  
  geom_sf(
    data = map_spatial,
    
    aes(color = error_cap),
    
    alpha = 0.65,
    
    size = 0.75
  ) +
  
  scale_color_gradient(
    low = "darkgreen",
    high = "red",
    na.value = NA
  ) +
  
  coord_sf(
    xlim = c(bbox_data["xmin"], bbox_data["xmax"]),
    ylim = c(bbox_data["ymin"], bbox_data["ymax"]),
    expand = FALSE
  ) +
  
  labs(
    title = "Spatial CV Errors",
    subtitle = "Leave-locality-out validation",
    color = "Absolute error"
  ) +
  
  theme_minimal()

# ============================================================
# MAPA UPZ CV
# ============================================================

p_map_upz <- ggplot() +
  
  geom_sf(
    data = localidades_map,
    fill = "gray97",
    color = "gray75",
    linewidth = 0.25
  ) +
  
  geom_sf(
    data = map_upz,
    
    aes(color = error_cap),
    
    alpha = 0.65,
    
    size = 0.75
  ) +
  
  scale_color_gradient(
    low = "darkgreen",
    high = "red",
    na.value = NA
  ) +
  
  coord_sf(
    xlim = c(bbox_data["xmin"], bbox_data["xmax"]),
    ylim = c(bbox_data["ymin"], bbox_data["ymax"]),
    expand = FALSE
  ) +
  
  labs(
    title = "UPZ Spatial CV Errors",
    subtitle = "Leave-UPZ-out validation",
    color = "Absolute error"
  ) +
  
  theme_minimal()

# ============================================================
# MAPA NORTE
# ============================================================

p_map_norte <- ggplot() +
  
  geom_sf(
    data = localidades_map,
    fill = "gray97",
    color = "gray75",
    linewidth = 0.25
  ) +
  
  geom_sf(
    data = map_norte,
    
    aes(color = error_cap),
    
    alpha = 0.65,
    
    size = 0.75
  ) +
  
  scale_color_gradient(
    low = "darkgreen",
    high = "red",
    na.value = NA
  ) +
  
  coord_sf(
    xlim = c(bbox_data["xmin"], bbox_data["xmax"]),
    ylim = c(bbox_data["ymin"], bbox_data["ymax"]),
    expand = FALSE
  ) +
  
  labs(
    title = "Northeast Cluster Holdout Errors",
    subtitle = "Most realistic Kaggle proxy",
    color = "Absolute error"
  ) +
  
  theme_minimal()

# ============================================================
# EXPORTAR
# ============================================================

ggsave(
  here(paths$figures, "map_error_spatial.png"),
  p_map_spatial,
  width = 8,
  height = 8
)

ggsave(
  here(paths$figures, "map_error_upz.png"),
  p_map_upz,
  width = 8,
  height = 8
)

ggsave(
  here(paths$figures, "map_error_norte.png"),
  p_map_norte,
  width = 8,
  height = 8
)

# ============================================================
# ERROR BY LOCALIDAD
# ============================================================

localidad_error_map <- train_sf |>
  
  st_drop_geometry() |>
  
  mutate(
    error = abs(cv_spatial$pred - price)
  ) |>
  
  group_by(LocNombre) |>
  
  summarise(
    mean_error = mean(error, na.rm = TRUE),
    median_error = median(error, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(
  localidad_error_map,
  here(paths$tables, "localidad_error_summary.csv")
)

# ============================================================
# ERROR VS DISTANCE
# ============================================================

if ("dist_centro" %in% names(train_sf)) {
  
  dist_df <- tibble(
    dist = train_sf$dist_centro,
    err  = abs(cv_spatial$pred - y_train)
  )
  
  p_dist <- ggplot(dist_df, aes(dist, err)) +
    geom_point(alpha = 0.1) +
    geom_smooth(se = FALSE) +
    labs(
      title = "Error vs Distance",
      x = "Distance",
      y = "Absolute Error"
    )
  
  ggsave(
    here(paths$figures, "error_vs_distance.png"),
    p_dist,
    width = 8,
    height = 6
  )
}

# ============================================================
# EXPORTS
# ============================================================

saveRDS(
  fit_final,
  here(paths$training, "xgb_best_model_analysis.rds")
)

saveRDS(
  tabla_cv,
  here(paths$training, "tabla_cv_analysis.rds")
)

saveRDS(
  bias_tbl,
  here(paths$training, "bias_tbl_analysis.rds")
)

saveRDS(
  benchmark_tbl,
  here(paths$training, "benchmark_tbl.rds")
)

message("======================================================")
message("10_analysis.R COMPLETADO")
message("======================================================")
message("Outputs generados:")
message("  • CV comparison tables")
message("  • Fold-level MAE")
message("  • Spatial validation gap")
message("  • XGBoost gain importance")
message("  • SHAP analysis")
message("  • Bias analysis")
message("  • Catastrophic errors")
message("  • Benchmark comparison")
message("  • Spatial error maps")
message("======================================================")