# ============================================================
# 04_cv_setup.R
# Construcción de folds con tidymodels (rsample)
# ============================================================

#   Chapinero tiene pocas obs en train → se comporta como zona
#   parcialmente nueva. La CV aleatoria subestimaría el error
#   porque Cov(ε_i, ε_j) ≠ 0 para obs cercanas (leakage).
#   La CV espacial (leave-location-out) fuerza ρ ≈ 0 entre
#   train y validation → estimación más honesta del error real.
#
#   Además, zonas del noreste de Bogotá (Usaquén, Teusaquillo)
#   son más parecidas a Chapinero en precio y características
#   que otras localidades → se genera un fold "cluster norte"
#   que simula exactamente el escenario del Kaggle.
#
# Folds producidos:
#
#   folds_std          — k=5 aleatorio
#   folds_global       — k=5 estratificado por localidad
#   folds_spatial      — leave-location-out por localidad
#                        → CV espacial del profesor: ρ ≈ 0
#   folds_spatial_upz  — leave-UPZ-out (más granular)
#   folds_norte        — split: train=sur/centro Bogotá
#                        test=cluster norte (Chapinero + Usaquén
#                        + Teusaquillo + Barrios Unidos)
#                        → mejor proxy del Kaggle: simula predecir
#                          en Chapinero con apoyo de zonas similares
#   folds_pca          — k=5 estratificado sobre train_pca_sf
#
# Pesos por distancia a Chapinero:
#   Se calculan y guardan para usarse en SL y modelos finales
#   → obs cercanas a Chapinero reciben más peso en el entrenamiento
#   → implementa la intuición de autocorrelación espacial del profe
#
# ============================================================

p_load(tidymodels, rsample)

# --- Cargar ------------------------------------------------------------------

train     <- readRDS(here(paths$processed, "train_model.rds"))
train_pca <- readRDS(here(paths$processed, "train_pca.rds"))

# --- Convertir a sf y agregar log_price --------------------------------------

train_sf <- train |>
  st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) |>
  mutate(log_price = log(price))

train_pca_sf <- train_pca |>
  st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) |>
  mutate(log_price = log(price))

# Guardar versiones sf
saveRDS(train_sf,     here(paths$processed, "train_sf.rds"))
saveRDS(train_pca_sf, here(paths$processed, "train_pca_sf.rds"))

# =============================================================
# DIAGNÓSTICO: Chapinero en train
# =============================================================

dist_loc <- train_sf |>
  st_drop_geometry() |>
  count(LocCodigo, LocNombre) |>
  arrange(desc(n))

message("\nDistribución por localidad:")
print(dist_loc)

n_chap     <- dist_loc |> filter(LocNombre == "CHAPINERO") |> pull(n)
pct_chap   <- round(n_chap / nrow(train_sf) * 100, 1)
message("\nChapinero en train: ", n_chap, " obs (", pct_chap, "% del total)")

if (n_chap < 200) {
  message("ALERTA: Chapinero tiene pocas obs → se comporta como zona parcialmente nueva")
  message("  → folds_spatial y folds_norte son los proxies correctos del Kaggle score")
} else {
  message("Chapinero tiene obs suficientes → folds_norte sigue siendo el mejor proxy")
}

# =============================================================
# 1. CV ESTÁNDAR — k = 5 aleatorio
# =============================================================

set.seed(SEED)
folds_std <- vfold_cv(train_sf, v = CV_FOLDS, strata = NULL)
message("\nCV estándar  |  ", CV_FOLDS, " folds")

# =============================================================
# 2. CV GLOBAL — k = 5 estratificado por localidad
# =============================================================

set.seed(SEED)
folds_global <- vfold_cv(train_sf, v = CV_FOLDS, strata = LocCodigo)
message("CV global    |  ", CV_FOLDS, " folds (estratificado por localidad)")

# =============================================================
# 3. CV ESPACIAL — leave-location-out por localidad
# =============================================================
# Lecture 13: fuerza ρ ≈ 0 entre train y validation
# → estimación honesta del error de generalización

set.seed(SEED)
folds_spatial <- group_vfold_cv(train_sf, group = LocCodigo)
message("CV espacial  |  ", nrow(folds_spatial), " folds (leave-location-out)")

# =============================================================
# 4. CV ESPACIAL — leave-UPZ-out
# =============================================================

set.seed(SEED)
folds_spatial_upz <- group_vfold_cv(train_sf, group = CODIGO_UPZ)
message("CV UPZ       |  ", nrow(folds_spatial_upz), " folds (leave-UPZ-out)")

# =============================================================
# 5. CLUSTER NORTE — mejor proxy del Kaggle score
# =============================================================
# Chapinero tiene pocas obs → se comporta como zona nueva.
# Zonas del noreste son las más similares en precio y características
# (mismo nivel socioeconómico, mismo tipo de mercado inmobiliario):
#   - Usaquén    → norte, estratos 4-6, similar a Chapinero
#   - Teusaquillo→ centro-norte, también estrato 4-5
#   - Barrios Unidos → contiguo a Chapinero
#
# Este fold simula: "entreno en sur/centro/occidente Bogotá,
# predigo en el cluster norte donde está el test de Kaggle"
# → es el proxy más realista del escenario Kaggle

localidades_norte <- c(
  "CHAPINERO",
  "USAQUEN",
  "TEUSAQUILLO",
  "BARRIOS UNIDOS"
)

idx_norte <- which(train_sf$LocNombre %in% localidades_norte)
idx_sur   <- which(!train_sf$LocNombre %in% localidades_norte)

n_norte <- length(idx_norte)
n_sur   <- length(idx_sur)

message("\nCluster norte (Chapinero + Usaquén + Teusaquillo + Barrios Unidos):")
train_sf |>
  st_drop_geometry() |>
  filter(LocNombre %in% localidades_norte) |>
  count(LocNombre) |>
  arrange(desc(n)) |>
  print()

message("  Total cluster norte: ", n_norte, " obs (",
        round(n_norte / nrow(train_sf) * 100, 1), "% del total)")
message("  Train (resto):       ", n_sur, " obs")

split_norte <- make_splits(
  x    = list(analysis   = idx_sur,
              assessment = idx_norte),
  data = train_sf
)

folds_norte <- manual_rset(
  splits = list(split_norte),
  ids    = "Cluster_Norte"
)

message("CV norte     |  1 split (cluster norte como test)")

# =============================================================
# 6. CV PCA — k=5 estratificado, sobre train_pca_sf
# =============================================================

set.seed(SEED)
folds_pca <- vfold_cv(train_pca_sf, v = CV_FOLDS, strata = LocCodigo)
message("CV PCA       |  ", CV_FOLDS, " folds (estratificado por localidad)")

# =============================================================
# PESOS POR DISTANCIA A CHAPINERO
# =============================================================
# Lecture 13: la autocorrelación espacial implica que obs cercanas
# a la zona de predicción son más informativas.
# → al entrenar el modelo final, dar más peso a obs del cluster norte
#
# Estrategia:
#   1. Calcular distancia de cada obs al centroide de Chapinero
#   2. Transformar a peso usando kernel gaussiano:
#      w_i = exp(-d_i² / (2 * bandwidth²))
#      → obs en Chapinero: peso ≈ 1
#      → obs lejanas (sur de Bogotá): peso → 0
#   3. Normalizar para que los pesos sumen n_train (convención)
#
# bandwidth: radio en metros donde el peso cae a ~0.6
#   Usamos 3km → cubre Chapinero + vecinos inmediatos
#   Usamos también 6km → cubre cluster norte completo
# =============================================================

# Proyectar a sistema métrico para calcular distancias en metros
train_m <- train_sf |> st_transform(3116)

# Centroide de Chapinero (calculado desde las obs de train)
centroide_chapinero <- train_m |>
  filter(LocNombre == "CHAPINERO") |>
  st_union() |>
  st_centroid()

# Distancia de cada obs al centroide de Chapinero (metros)
dist_a_chapinero <- st_distance(train_m, centroide_chapinero) |>
  as.numeric()

message("\nDistancias al centroide de Chapinero:")
message("  min: ",  round(min(dist_a_chapinero) / 1000, 2), " km")
message("  mediana: ", round(median(dist_a_chapinero) / 1000, 2), " km")
message("  max: ",  round(max(dist_a_chapinero) / 1000, 2), " km")

# Kernel gaussiano con dos bandwidths
bw_3km  <- 3000   # 3 km → focalizado en Chapinero y vecinos inmediatos
bw_6km  <- 6000   # 6 km → cluster norte completo

pesos_gauss_3km <- exp(-(dist_a_chapinero^2) / (2 * bw_3km^2))
pesos_gauss_6km <- exp(-(dist_a_chapinero^2) / (2 * bw_6km^2))

# Normalizar: pesos suman n_train (convención para obsWeights en glmnet/ranger)
n_train <- nrow(train_sf)
pesos_3km <- pesos_gauss_3km / mean(pesos_gauss_3km)
pesos_6km <- pesos_gauss_6km / mean(pesos_gauss_6km)

# Pesos uniformes (baseline)
pesos_uniform <- rep(1, n_train)

message("\nResumen pesos gaussianos (3km bandwidth):")
message("  mean: ", round(mean(pesos_3km), 3),
        "  min: ", round(min(pesos_3km), 4),
        "  max: ", round(max(pesos_3km), 2))
message("  Obs del cluster norte reciben peso promedio: ",
        round(mean(pesos_3km[idx_norte]), 2))
message("  Obs fuera del cluster reciben peso promedio: ",
        round(mean(pesos_3km[idx_sur]), 3))

# Verificar coherencia: las obs de Chapinero deben tener los pesos más altos
top_pesos <- train_sf |>
  st_drop_geometry() |>
  mutate(peso_3km = pesos_3km) |>
  group_by(LocNombre) |>
  summarise(peso_medio = round(mean(peso_3km), 3)) |>
  arrange(desc(peso_medio))

message("\nPeso medio por localidad (bandwidth 3km):")
print(top_pesos)

# =============================================================
# GUARDAR
# =============================================================

saveRDS(folds_std,         here(paths$processed, "folds_std.rds"))
saveRDS(folds_global,      here(paths$processed, "folds_global.rds"))
saveRDS(folds_spatial,     here(paths$processed, "folds_spatial.rds"))
saveRDS(folds_spatial_upz, here(paths$processed, "folds_spatial_upz.rds"))
saveRDS(folds_norte,       here(paths$processed, "folds_norte.rds"))
saveRDS(folds_pca,         here(paths$processed, "folds_pca.rds"))

# Pesos guardados como vectores (usados en SL y modelos finales)
saveRDS(pesos_3km,         here(paths$processed, "pesos_dist_3km.rds"))
saveRDS(pesos_6km,         here(paths$processed, "pesos_dist_6km.rds"))
saveRDS(pesos_uniform,     here(paths$processed, "pesos_uniform.rds"))
saveRDS(dist_a_chapinero,  here(paths$processed, "dist_a_chapinero.rds"))

message("\n04_cv_setup.R")
message("Resumen de folds disponibles:")
message("  folds_std          → k=5 aleatorio")
message("  folds_global       → k=5 estratificado por localidad")
message("  folds_spatial      → leave-location-out (CV espacial del profe)")
message("  folds_spatial_upz  → leave-UPZ-out (más granular)")
message("  folds_norte        → cluster norte como test (mejor proxy Kaggle)")
message("  folds_pca          → k=5 estratificado para dataset PCA")
message("  pesos_3km / pesos_6km → kernel gaussiano centrado en Chapinero")
