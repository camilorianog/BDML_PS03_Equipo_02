# ============================================================
# 03_random_forest_refined.R
# Random Forest refinado sobre train_model.rds / test_model.rds
# Problem Set 03 - Housing Prices
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
  glue
)

set.seed(202604)

# ============================================================
# 1. RUTAS
# ============================================================

path_processed <- here::here("00_data", "01_processed")
path_competition <- here::here("00_data", "00_raw", "00_competition")
path_outputs <- here::here("02_models", "01_submissions", "03_random_forest")

dir.create(path_outputs, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 2. CARGAR BASES PROCESADAS
# ============================================================

train_model <- readRDS(file.path(path_processed, "train_model.rds")) |>
  janitor::clean_names()

test_model <- readRDS(file.path(path_processed, "test_model.rds")) |>
  janitor::clean_names()

message("Train cargado: ", nrow(train_model), " filas")
message("Test cargado: ", nrow(test_model), " filas")

# Submission template, si existe
submission_template_path <- file.path(path_competition, "submission_template.csv")

if (file.exists(submission_template_path)) {
  submission_template <- read.csv(submission_template_path) |>
    janitor::clean_names()
} else {
  submission_template <- test_model |>
    select(property_id)
}

# ============================================================
# 2.1 ELIMINAR COLUMNAS ESPACIALES TIPO sf / sfc
# ============================================================

drop_sfc_columns <- function(df) {
  
  sfc_cols <- names(df)[sapply(df, function(x) inherits(x, "sfc"))]
  sf_cols  <- names(df)[sapply(df, function(x) inherits(x, "sf"))]
  
  cols_to_drop <- unique(c(sfc_cols, sf_cols))
  
  if (length(cols_to_drop) > 0) {
    message("Eliminando columnas espaciales: ", paste(cols_to_drop, collapse = ", "))
    df <- df |>
      select(-all_of(cols_to_drop))
  }
  
  # Si el dataframe completo hereda clase sf, quitar geometría activa
  if (inherits(df, "sf")) {
    df <- sf::st_drop_geometry(df)
  }
  
  df <- as.data.frame(df)
  
  return(df)
}

train_model <- drop_sfc_columns(train_model)
test_model  <- drop_sfc_columns(test_model)

# ============================================================
# 3. DETECTAR VARIABLE OBJETIVO
# ============================================================

# Prioridad:
# 1. Si existe log_price, usamos log_price.
# 2. Si existe price, creamos log_price = log(price).

if ("log_price" %in% names(train_model)) {
  
  target_var <- "log_price"
  message("Usando variable objetivo existente: log_price")
  
} else if ("price" %in% names(train_model)) {
  
  train_model <- train_model |>
    filter(!is.na(price), price > 0) |>
    mutate(log_price = log(price))
  
  target_var <- "log_price"
  message("Creando log_price a partir de price")
  
} else {
  
  stop("No encuentro ni log_price ni price en train_model.rds")
}

# ============================================================
# 4. LIMPIEZA LIGERA
# ============================================================

# Guardar IDs del test
if (!"property_id" %in% names(test_model)) {
  stop("test_model no tiene property_id. Se necesita para crear la submission.")
}

test_ids <- test_model$property_id

# Columnas que no deben entrar al modelo
cols_remove <- c(
  "price",
  "property_id",
  "title",
  "description",
  "title_clean",
  "description_clean",
  "text_all",
  "geometry"
)

# Asegurar que train y test tengan las mismas columnas predictoras
predictor_cols <- setdiff(names(train_model), c(cols_remove, target_var))
predictor_cols <- intersect(predictor_cols, names(test_model))

train_rf <- train_model |>
  select(all_of(target_var), all_of(predictor_cols)) |>
  rename(y = all_of(target_var))

test_rf <- test_model |>
  select(all_of(predictor_cols))

# Convertir listas o columnas raras a character
train_rf <- train_rf |>
  mutate(across(where(is.list), as.character))

test_rf <- test_rf |>
  mutate(across(where(is.list), as.character))

# Convertir character a factor
train_rf <- train_rf |>
  mutate(across(where(is.character), as.factor))

test_rf <- test_rf |>
  mutate(across(where(is.character), as.factor))

# Alinear niveles de factores entre train y test
combined_x <- bind_rows(
  train_rf |> select(-y) |> mutate(.dataset = "train"),
  test_rf |> mutate(.dataset = "test")
)

combined_x <- combined_x |>
  mutate(across(where(is.character), as.factor))

# Imputación simple y segura
for (v in names(combined_x)) {
  
  if (v == ".dataset") next
  
  if (is.numeric(combined_x[[v]])) {
    
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

train_rf_clean <- bind_cols(
  tibble(y = train_rf$y),
  x_train
)

test_rf_clean <- x_test

message("Número de predictores usados: ", ncol(test_rf_clean))

# ============================================================
# 5. SPLIT DE VALIDACIÓN
# ============================================================

set.seed(202604)

n <- nrow(train_rf_clean)

valid_idx <- sample(seq_len(n), size = floor(0.20 * n))

train_idx <- setdiff(seq_len(n), valid_idx)

rf_train <- train_rf_clean[train_idx, ]
rf_valid <- train_rf_clean[valid_idx, ]

message("Train interno: ", nrow(rf_train))
message("Validación interna: ", nrow(rf_valid))

# ============================================================
# 6. GRID PEQUEÑO RANDOM FOREST
# ============================================================

p <- ncol(rf_train) - 1

grid_rf <- tibble(
  model_id = c(
    "rf_mtry_sqrt_min5",
    "rf_mtry_sqrt_min10",
    "rf_mtry_30pct_min5",
    "rf_mtry_30pct_min10",
    "rf_mtry_50pct_min10"
  ),
  mtry = c(
    max(2, floor(sqrt(p))),
    max(2, floor(sqrt(p))),
    max(2, floor(0.30 * p)),
    max(2, floor(0.30 * p)),
    max(2, floor(0.50 * p))
  ),
  min_node_size = c(5, 10, 5, 10, 10),
  sample_fraction = c(0.80, 0.80, 0.80, 0.80, 0.85)
)

print(grid_rf)

# ============================================================
# 7. ENTRENAR Y VALIDAR MODELOS LIVIANOS
# ============================================================

results <- list()
models <- list()

for (i in seq_len(nrow(grid_rf))) {
  
  cfg <- grid_rf[i, ]
  
  message("Entrenando: ", cfg$model_id)
  
  rf_fit <- ranger(
    formula = y ~ .,
    data = rf_train,
    num.trees = 500,
    mtry = cfg$mtry,
    min.node.size = cfg$min_node_size,
    sample.fraction = cfg$sample_fraction,
    importance = "impurity",
    respect.unordered.factors = "order",
    num.threads = max(1, parallel::detectCores() - 1),
    seed = 202604
  )
  
  pred_valid_log <- predict(rf_fit, data = rf_valid)$predictions
  
  mae_log <- mean(abs(rf_valid$y - pred_valid_log), na.rm = TRUE)
  mae_price <- mean(abs(exp(rf_valid$y) - exp(pred_valid_log)), na.rm = TRUE)
  
  results[[i]] <- tibble(
    model_id = cfg$model_id,
    mtry = cfg$mtry,
    min_node_size = cfg$min_node_size,
    sample_fraction = cfg$sample_fraction,
    mae_log = mae_log,
    mae_price = mae_price
  )
  
  models[[cfg$model_id]] <- rf_fit
  
  message("MAE validación precio: ", round(mae_price, 2))
}

rf_results <- bind_rows(results) |>
  arrange(mae_price)

write.csv(
  rf_results,
  file.path(path_outputs, "rf_validation_results.csv"),
  row.names = FALSE
)

message("Resultados validación:")
print(rf_results)

best_model_id <- rf_results$model_id[1]
best_cfg <- grid_rf |>
  filter(model_id == best_model_id)

message("Mejor configuración: ", best_model_id)

# ============================================================
# 8. ENTRENAR MODELO FINAL CON TODO TRAIN
# ============================================================

message("Entrenando Random Forest final con todo el train...")

rf_final <- ranger(
  formula = y ~ .,
  data = train_rf_clean,
  num.trees = 700,
  mtry = best_cfg$mtry,
  min.node.size = best_cfg$min_node_size,
  sample.fraction = best_cfg$sample_fraction,
  importance = "impurity",
  respect.unordered.factors = "order",
  num.threads = max(1, parallel::detectCores() - 1),
  seed = 202604
)

# Variante un poco más suave para blend
message("Entrenando variante conservadora...")

rf_conservative <- ranger(
  formula = y ~ .,
  data = train_rf_clean,
  num.trees = 700,
  mtry = max(2, floor(sqrt(p))),
  min.node.size = 15,
  sample.fraction = 0.80,
  importance = "impurity",
  respect.unordered.factors = "order",
  num.threads = max(1, parallel::detectCores() - 1),
  seed = 202605
)

# ============================================================
# 9. PREDICCIONES
# ============================================================

pred_rf_log <- predict(rf_final, data = test_rf_clean)$predictions
pred_rf_cons_log <- predict(rf_conservative, data = test_rf_clean)$predictions

pred_rf <- exp(pred_rf_log)
pred_rf_cons <- exp(pred_rf_cons_log)

pred_blend_rf <- 0.80 * pred_rf + 0.20 * pred_rf_cons

# ============================================================
# 10. FUNCIÓN SUBMISSION
# ============================================================

make_submission <- function(pred_price, file_name) {
  
  sub <- submission_template |>
    mutate(
      price = as.numeric(pred_price),
      price = ifelse(is.na(price), median(price, na.rm = TRUE), price),
      price = pmax(price, 50000000)
    ) |>
    select(property_id, price)
  
  if (nrow(sub) != length(test_ids)) {
    warning("El número de filas de la submission no coincide con test_model.")
  }
  
  write.csv(
    sub,
    file.path(path_outputs, file_name),
    row.names = FALSE
  )
  
  message("Guardado: ", file.path(path_outputs, file_name))
}

# ============================================================
# 11. GUARDAR SUBMISSIONS
# ============================================================

make_submission(
  pred_rf,
  "RF_refined_best.csv"
)

make_submission(
  pred_rf * 0.995,
  "RF_refined_best_cali0995.csv"
)

make_submission(
  pred_rf * 1.005,
  "RF_refined_best_cali1005.csv"
)

make_submission(
  pred_rf_cons,
  "RF_refined_conservative.csv"
)

make_submission(
  pred_blend_rf,
  "BLEND_RF_best80_conservative20.csv"
)

# ============================================================
# 12. GUARDAR MODELOS E IMPORTANCIA
# ============================================================

saveRDS(
  rf_final,
  file.path(path_outputs, "rf_final_refined.rds")
)

saveRDS(
  rf_conservative,
  file.path(path_outputs, "rf_conservative_refined.rds")
)

importance <- tibble(
  variable = names(rf_final$variable.importance),
  importance = as.numeric(rf_final$variable.importance)
) |>
  arrange(desc(importance))

write.csv(
  importance,
  file.path(path_outputs, "rf_variable_importance.csv"),
  row.names = FALSE
)

message("============================================================")
message("Proceso terminado.")
message("Submissions guardadas en:")
message(path_outputs)
message("Mejor modelo según validación interna:")
message(best_model_id)
message("============================================================")