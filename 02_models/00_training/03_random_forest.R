# ============================================================
# 03_random_forest_competitive_v2_clean.R
# Random Forest competitivo - solo genera:
# RF_competitive_v2_lm_calibrated.csv
# ============================================================

# ============================================================
# 0. PAQUETES
# ============================================================

if (!require("pacman", quietly = TRUE)) install.packages("pacman")

pacman::p_load(
  tidyverse,
  janitor,
  ranger,
  here,
  forcats,
  sf
)

set.seed(202606)

# ============================================================
# 1. RUTAS
# ============================================================

path_processed <- here::here("00_data", "01_processed")
path_competition <- here::here("00_data", "00_raw", "00_competition")
path_outputs <- here::here("02_models", "01_submissions", "03_random_forest")

dir.create(path_outputs, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 2. CARGAR BASES
# ============================================================

train_model <- readRDS(file.path(path_processed, "train_model.rds")) |>
  janitor::clean_names()

test_model <- readRDS(file.path(path_processed, "test_model.rds")) |>
  janitor::clean_names()

submission_template <- read.csv(file.path(path_competition, "submission_template.csv")) |>
  janitor::clean_names()

message("Train cargado: ", nrow(train_model), " filas")
message("Test cargado: ", nrow(test_model), " filas")

# ============================================================
# 3. ELIMINAR COLUMNAS ESPACIALES / LISTAS
# ============================================================

drop_bad_columns <- function(df) {
  
  if (inherits(df, "sf")) {
    df <- sf::st_drop_geometry(df)
  }
  
  bad_cols <- names(df)[sapply(df, function(x) {
    inherits(x, "sfc") ||
      inherits(x, "sf") ||
      is.list(x)
  })]
  
  if (length(bad_cols) > 0) {
    message("Eliminando columnas incompatibles: ", paste(bad_cols, collapse = ", "))
    df <- df |> select(-all_of(bad_cols))
  }
  
  as.data.frame(df)
}

train_model <- drop_bad_columns(train_model)
test_model <- drop_bad_columns(test_model)

# ============================================================
# 4. VARIABLE OBJETIVO
# ============================================================

if ("log_price" %in% names(train_model)) {
  
  train_model <- train_model |>
    filter(!is.na(log_price))
  
  target_var <- "log_price"
  message("Usando log_price existente.")
  
} else if ("price" %in% names(train_model)) {
  
  train_model <- train_model |>
    filter(!is.na(price), price > 0) |>
    mutate(log_price = log(price))
  
  target_var <- "log_price"
  message("Creando log_price desde price.")
  
} else {
  stop("No encuentro price ni log_price en train_model.")
}

if ("price" %in% names(train_model)) {
  y_price_real <- train_model$price
} else {
  y_price_real <- exp(train_model[[target_var]])
}

# ============================================================
# 5. PREPARAR BASE PARA RANDOM FOREST
# ============================================================

if (!"property_id" %in% names(test_model)) {
  stop("test_model no tiene property_id.")
}

test_ids <- test_model$property_id

cols_remove <- c(
  "property_id",
  "price",
  "log_price",
  "title",
  "description",
  "title_clean",
  "description_clean",
  "text_all",
  "geometry"
)

predictor_cols <- setdiff(names(train_model), cols_remove)
predictor_cols <- intersect(predictor_cols, names(test_model))

train_x <- train_model |> select(all_of(predictor_cols))
test_x <- test_model |> select(all_of(predictor_cols))

combined_x <- bind_rows(
  train_x |> mutate(.dataset = "train"),
  test_x |> mutate(.dataset = "test")
)

bad_cols <- names(combined_x)[sapply(combined_x, function(x) {
  inherits(x, "sfc") || inherits(x, "sf") || is.list(x)
})]

if (length(bad_cols) > 0) {
  message("Eliminando columnas incompatibles restantes: ", paste(bad_cols, collapse = ", "))
  combined_x <- combined_x |> select(-all_of(bad_cols))
}

# Imputación y factores
for (v in names(combined_x)) {
  
  if (v == ".dataset") next
  
  if (is.numeric(combined_x[[v]]) || is.integer(combined_x[[v]])) {
    
    combined_x[[v]] <- as.numeric(combined_x[[v]])
    
    med_v <- median(combined_x[[v]], na.rm = TRUE)
    if (is.na(med_v) || is.nan(med_v)) med_v <- 0
    
    combined_x[[v]][is.na(combined_x[[v]])] <- med_v
    combined_x[[v]][is.infinite(combined_x[[v]])] <- med_v
    
  } else {
    
    combined_x[[v]] <- as.factor(combined_x[[v]])
    combined_x[[v]] <- forcats::fct_explicit_na(combined_x[[v]], na_level = "missing")
  }
}

x_train <- combined_x |>
  filter(.dataset == "train") |>
  select(-.dataset)

x_test <- combined_x |>
  filter(.dataset == "test") |>
  select(-.dataset)

train_rf <- bind_cols(
  tibble(
    y_log = train_model[[target_var]],
    y_price = y_price_real
  ),
  x_train
)

test_rf <- x_test

message("Predictores usados: ", ncol(test_rf))

# ============================================================
# 6. SPLIT DE VALIDACIÓN
# ============================================================

set.seed(202606)

n <- nrow(train_rf)
valid_idx <- sample(seq_len(n), size = floor(0.20 * n))
train_idx <- setdiff(seq_len(n), valid_idx)

rf_train <- train_rf[train_idx, ]
rf_valid <- train_rf[valid_idx, ]

p <- ncol(test_rf)

message("Train interno: ", nrow(rf_train))
message("Validación interna: ", nrow(rf_valid))
message("Número de predictores: ", p)

# ============================================================
# 7. GRID RANDOM FOREST
# ============================================================

grid_rf <- tibble(
  model_id = c(
    "rf_35pct_min3_sf085",
    "rf_45pct_min3_sf085",
    "rf_55pct_min5_sf085",
    "rf_65pct_min5_sf090",
    "rf_45pct_min8_sf090",
    "rf_55pct_min10_sf090"
  ),
  mtry = c(
    max(2, floor(0.35 * p)),
    max(2, floor(0.45 * p)),
    max(2, floor(0.55 * p)),
    max(2, floor(0.65 * p)),
    max(2, floor(0.45 * p)),
    max(2, floor(0.55 * p))
  ),
  min_node_size = c(3, 3, 5, 5, 8, 10),
  sample_fraction = c(0.85, 0.85, 0.85, 0.90, 0.90, 0.90),
  num_trees = c(700, 700, 700, 700, 700, 700)
)

print(grid_rf)

# ============================================================
# 8. VALIDACIÓN PARA ESCOGER MEJOR RF Y CALIBRACIÓN
# ============================================================

results <- list()

for (i in seq_len(nrow(grid_rf))) {
  
  cfg <- grid_rf[i, ]
  
  message("============================================================")
  message("Entrenando modelo: ", cfg$model_id)
  
  rf_fit <- ranger(
    formula = y_log ~ .,
    data = rf_train |> select(-y_price),
    num.trees = cfg$num_trees,
    mtry = cfg$mtry,
    min.node.size = cfg$min_node_size,
    sample.fraction = cfg$sample_fraction,
    splitrule = "variance",
    respect.unordered.factors = "order",
    importance = "none",
    num.threads = max(1, parallel::detectCores() - 1),
    seed = 202606 + i
  )
  
  pred_valid_log <- predict(
    rf_fit,
    data = rf_valid |> select(-y_log, -y_price)
  )$predictions
  
  pred_valid_price <- exp(pred_valid_log)
  
  mae_price <- mean(abs(rf_valid$y_price - pred_valid_price), na.rm = TRUE)
  
  calib_df <- tibble(
    y_log = rf_valid$y_log,
    pred_log = pred_valid_log
  )
  
  calib_lm <- lm(y_log ~ pred_log, data = calib_df)
  
  pred_valid_calib_log <- predict(calib_lm, newdata = calib_df)
  pred_valid_calib_price <- exp(pred_valid_calib_log)
  
  mae_price_calib <- mean(abs(rf_valid$y_price - pred_valid_calib_price), na.rm = TRUE)
  
  results[[i]] <- tibble(
    model_id = cfg$model_id,
    mtry = cfg$mtry,
    min_node_size = cfg$min_node_size,
    sample_fraction = cfg$sample_fraction,
    num_trees = cfg$num_trees,
    mae_price = mae_price,
    mae_price_calib = mae_price_calib,
    calib_intercept = coef(calib_lm)[1],
    calib_slope = coef(calib_lm)[2]
  )
  
  message("MAE precio raw:   ", round(mae_price, 2))
  message("MAE precio calib: ", round(mae_price_calib, 2))
}

rf_results <- bind_rows(results) |>
  arrange(mae_price_calib)

print(rf_results)

best_model_id <- rf_results$model_id[1]
best_cfg <- grid_rf |> filter(model_id == best_model_id)

best_calib_intercept <- rf_results$calib_intercept[1]
best_calib_slope <- rf_results$calib_slope[1]

message("============================================================")
message("Mejor modelo por MAE calibrado: ", best_model_id)
message("Intercept calibración: ", best_calib_intercept)
message("Slope calibración: ", best_calib_slope)
message("============================================================")

# ============================================================
# 9. ENTRENAR MODELO FINAL CON TODO EL TRAIN
# ============================================================

message("Entrenando RF final con todo el train...")

rf_final <- ranger(
  formula = y_log ~ .,
  data = train_rf |> select(-y_price),
  num.trees = 1000,
  mtry = best_cfg$mtry,
  min.node.size = best_cfg$min_node_size,
  sample.fraction = best_cfg$sample_fraction,
  splitrule = "variance",
  respect.unordered.factors = "order",
  importance = "none",
  num.threads = max(1, parallel::detectCores() - 1),
  seed = 202700
)

# ============================================================
# 10. PREDICCIÓN CALIBRADA
# ============================================================

pred_final_log <- predict(rf_final, data = test_rf)$predictions

pred_final_calibrated <- exp(
  best_calib_intercept + best_calib_slope * pred_final_log
)

# ============================================================
# 11. GUARDAR ÚNICA SUBMISSION
# ============================================================

submission <- submission_template |>
  mutate(
    price = as.numeric(pred_final_calibrated),
    price = ifelse(is.na(price), median(price, na.rm = TRUE), price),
    price = pmax(price, 50000000)
  ) |>
  select(property_id, price)

if (nrow(submission) != length(test_ids)) {
  warning("El número de filas de la submission no coincide con test_model.")
}

output_file <- file.path(path_outputs, "RF_competitive_v2_lm_calibrated.csv")

write.csv(
  submission,
  output_file,
  row.names = FALSE
)

message("============================================================")
message("Proceso terminado.")
message("Única submission guardada en:")
message(output_file)
message("============================================================")
