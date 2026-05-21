# ============================================================
# 04_cv_setup.R
# Construcción de folds con tidymodels (rsample)
# ============================================================
# Team 02 — Problem Set 03
#
# Produce CUATRO objetos rset listos para tidymodels:
#
#   folds_std          — k=5 aleatorio
#   folds_global       — k=5 estratificado por localidad
#   folds_spatial      — leave-location-out por localidad
#   folds_spatial_upz  — leave-UPZ-out por UPZ
#
# ============================================================

p_load(tidymodels, rsample)

# --- Cargar ------------------------------------------------------------------

if (exists("train", envir = .GlobalEnv)) {
  train <- get("train", envir = .GlobalEnv)
  message("04_cv_setup.R  |  usando train en memoria")
} else if (file.exists(here(paths$processed, "train_model.rds"))) {
  train <- readRDS(here(paths$processed, "train_model.rds"))
  message("04_cv_setup.R  |  cargando train_model.rds")
} else {
  stop(
    "No hay 'train' en memoria ni train_model.rds. Ejecuta 03_imputation.R primero.",
    call. = FALSE
  )
}

# Reconvertir a sf
train_sf <- train |>
  st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE)

# Agregar log_price
train_sf <- train_sf |>
  mutate(log_price = log(price))

# Guardar versión sf
saveRDS(train_sf, here(paths$processed, "train_sf.rds"))

# =============================================================
# 1. CV ESTÁNDAR — k = 5 aleatorio
# =============================================================

set.seed(SEED)

folds_std <- vfold_cv(
  train_sf,
  v = CV_FOLDS,
  strata = NULL
)

message("CV estándar  |  ", CV_FOLDS, " folds")

# =============================================================
# 2. CV GLOBAL — k = 5 estratificado por localidad
# =============================================================
# Mezcla localidades en todos los folds
# Mantiene proporciones similares por localidad

set.seed(SEED)

folds_global <- vfold_cv(
  train_sf,
  v = CV_FOLDS,
  strata = LocCodigo
)

message("CV global    |  ",
        CV_FOLDS,
        " folds (estratificado por localidad)")

# =============================================================
# 3. CV ESPACIAL — leave-location-out por localidad
# =============================================================
# group_vfold_cv agrupa por LocCodigo y deja una localidad
# fuera en cada fold → simula predecir en zona desconocida

set.seed(SEED)

folds_spatial <- group_vfold_cv(
  train_sf,
  group = LocCodigo
)

message("CV espacial  |  ",
        nrow(folds_spatial),
        " folds (leave-location-out)")

# =============================================================
# 4. CV ESPACIAL — leave-UPZ-out
# =============================================================
# Más granular que localidad
# Evaluación espacial más exigente

set.seed(SEED)

folds_spatial_upz <- group_vfold_cv(
  train_sf,
  group = CODIGO_UPZ
)

message("CV UPZ       |  ",
        nrow(folds_spatial_upz),
        " folds (leave-UPZ-out)")

# =============================================================
# VERIFICACIÓN
# =============================================================

# Distribución de obs por localidad
dist_loc <- train_sf |>
  st_drop_geometry() |>
  count(LocCodigo, LocNombre) |>
  arrange(desc(n))

message("\nDistribución por localidad:")
print(dist_loc)

message("\nChapinero en train: ",
        dist_loc |> filter(LocNombre == "CHAPINERO") |> pull(n),
        " obs (",
        round(
          dist_loc |>
            filter(LocNombre == "CHAPINERO") |>
            pull(n) / nrow(train_sf) * 100,
          1
        ),
        "% del total)")

# Distribución UPZ
dist_upz <- train_sf |>
  st_drop_geometry() |>
  count(CODIGO_UPZ) |>
  arrange(desc(n))

message("\nNúmero de UPZ: ", nrow(dist_upz))

# =============================================================
# GUARDAR
# =============================================================

saveRDS(folds_std,
        here(paths$processed, "folds_std.rds"))

saveRDS(folds_global,
        here(paths$processed, "folds_global.rds"))

saveRDS(folds_spatial,
        here(paths$processed, "folds_spatial.rds"))

saveRDS(folds_spatial_upz,
        here(paths$processed, "folds_spatial_upz.rds"))

message("\n04_cv_setup.R")