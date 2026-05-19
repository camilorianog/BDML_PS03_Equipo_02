# ============================================================
# 04_cv_setup.R
# Construcción de folds con tidymodels (rsample)
# ============================================================
# Team 02 — Problem Set 03
#
# Produce DOS objetos rset listos para tidymodels:
#
#   folds_std     — k=5 aleatorio (baseline estándar)
#   folds_spatial — leave-location-out por localidad
#                   usando group_vfold_cv() de rsample
#                   un fold por cada localidad de Bogotá
#
# ============================================================

p_load(tidymodels, rsample)

# --- Cargar ------------------------------------------------------------------

train <- readRDS(here(paths$processed, "train_model.rds"))

# Reconvertir a sf
train_sf <- train |>
  st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE)

# Guardar versión sf
saveRDS(train_sf, here(paths$processed, "train_sf.rds"))

# =============================================================
# 1. CV ESTÁNDAR — k = 5 aleatorio
# =============================================================

set.seed(SEED)

folds_std <- vfold_cv(train_sf, v = CV_FOLDS, strata = NULL)

message("CV estándar  |  ", CV_FOLDS, " folds")

# =============================================================
# 2. CV ESPACIAL — leave-location-out por localidad
# =============================================================
# group_vfold_cv agrupa por LocCodigo y deja una localidad
# fuera en cada fold → simula predecir en zona desconocida

set.seed(SEED)

folds_spatial <- group_vfold_cv(
  train_sf,
  group = LocCodigo
)

message("CV espacial  |  ", nrow(folds_spatial), " folds (leave-location-out)")

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
        round(dist_loc |> filter(LocNombre == "CHAPINERO") |> pull(n) / nrow(train_sf) * 100, 1),
        "% del total)")

# =============================================================
# GUARDAR
# =============================================================

saveRDS(folds_std,     here(paths$processed, "folds_std.rds"))
saveRDS(folds_spatial, here(paths$processed, "folds_spatial.rds"))

message("\n04_cv_setup.R")