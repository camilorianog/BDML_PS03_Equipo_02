# ============================================================
# 02_spatial_variables.R
# Variables espaciales derivadas de OpenStreetMap
# ============================================================
# Team 02 — Problem Set 03
# Variables construidas:
#   DISTANCIAS al punto más cercano (metros):
#     dist_cafe, dist_bus, dist_metro, dist_hospital,
#     dist_colegio, dist_universidad, dist_parque, dist_super
#   DENSIDADES conteo en buffer:
#     n_cafes_500m, n_rest_500m, n_farmacias_500m,
#     n_bancos_500m, n_gym_500m, n_lamparas_200m
#   BINARIAS:
#     is_residential
#   ADMINISTRATIVAS (join espacial):
#     ESTRATO, CODIGO_UPZ, NOMBRE_UPZ, LocCodigo, LocNombre
# ============================================================

# --- Helpers -----------------------------------------------------------------

#' Descarga puntos OSM para key = value en bbox de Bogotá
get_osm_points <- function(key, value, bbox = getbb("Bogota Colombia")) {
  message("  OSM: ", key, " = ", value)
  q   <- opq(bbox = bbox) |> add_osm_feature(key = key, value = value)
  raw <- osmdata_sf(q)
  raw$osm_points |> dplyr::select(osm_id) |> st_as_sf(crs = 4326)
}

#' Distancia mínima (metros) desde cada fila de base_sf a target_sf
dist_min <- function(base_sf, target_sf) {
  as.numeric(apply(st_distance(base_sf, target_sf), 1, min))
}

#' Conteo de features OSM dentro de un buffer (radio en metros)
count_buffer <- function(base_m, target_sf, radio = 500) {
  target_m <- st_transform(target_sf, 3116)
  buf      <- st_buffer(base_m, dist = radio)
  lengths(st_intersects(buf, target_m))
}

# --- Convertir a sf ----------------------------------------------------------

train <- train |> st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE)
test  <- test  |> st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE)

# Proyección métrica MAGNA-SIRGAS Colombia Bogotá (para buffers)
train_m <- train |> st_transform(3116)
test_m  <- test  |> st_transform(3116)

# =============================================================
# BLOQUE A — DISTANCIAS AL PUNTO MÁS CERCANO
# =============================================================

# A1. Cafés -------------------------------------------------------------------
cafes_sf        <- get_osm_points("amenity", "cafe")
train$dist_cafe <- dist_min(train, cafes_sf)
test$dist_cafe  <- dist_min(test,  cafes_sf)

# A2. Estaciones de bus -------------------------------------------------------
bus_sf          <- get_osm_points("amenity", "bus_station")
train$dist_bus  <- dist_min(train, bus_sf)
test$dist_bus   <- dist_min(test,  bus_sf)

# A3. Metro / cable -----------------------------------------------------------
#     Chapinero tiene estaciones del metro de Bogotá → premium importante
metro_sf           <- get_osm_points("railway", "station")
train$dist_metro   <- dist_min(train, metro_sf)
test$dist_metro    <- dist_min(test,  metro_sf)

# A4. Hospitales / clínicas ---------------------------------------------------
hospital_sf          <- get_osm_points("amenity", "hospital")
train$dist_hospital  <- dist_min(train, hospital_sf)
test$dist_hospital   <- dist_min(test,  hospital_sf)

# A5. Colegios ----------------------------------------------------------------
colegio_sf          <- get_osm_points("amenity", "school")
train$dist_colegio  <- dist_min(train, colegio_sf)
test$dist_colegio   <- dist_min(test,  colegio_sf)

# A6. Universidades -----------------------------------------------------------
#     Chapinero: Uniandes, Javeriana, Rosario, EAN, etc.
univ_sf                <- get_osm_points("amenity", "university")
train$dist_universidad <- dist_min(train, univ_sf)
test$dist_universidad  <- dist_min(test,  univ_sf)

# A7. Parques -----------------------------------------------------------------
#     Polígonos → centroide como proxy de acceso
q_parques   <- opq(bbox = getbb("Bogota Colombia")) |>
  add_osm_feature(key = "leisure", value = "park")
parques_sf  <- osmdata_sf(q_parques)$osm_polygons |>
  dplyr::select(osm_id) |>
  st_centroid()

train$dist_parque <- dist_min(train, parques_sf)
test$dist_parque  <- dist_min(test,  parques_sf)

# A8. Supermercados -----------------------------------------------------------
super_sf         <- get_osm_points("shop", "supermarket")
train$dist_super <- dist_min(train, super_sf)
test$dist_super  <- dist_min(test,  super_sf)

# =============================================================
# BLOQUE B — DENSIDADES EN BUFFER
# =============================================================

# B1. Cafés en radio de 500 m -------------------------------------------------
train$n_cafes_500m <- count_buffer(train_m, cafes_sf, 500)
test$n_cafes_500m  <- count_buffer(test_m,  cafes_sf, 500)

# B2. Restaurantes en 500 m ---------------------------------------------------
rest_sf           <- get_osm_points("amenity", "restaurant")
train$n_rest_500m <- count_buffer(train_m, rest_sf, 500)
test$n_rest_500m  <- count_buffer(test_m,  rest_sf, 500)

# B3. Farmacias en 500 m ------------------------------------------------------
farm_sf                <- get_osm_points("amenity", "pharmacy")
train$n_farmacias_500m <- count_buffer(train_m, farm_sf, 500)
test$n_farmacias_500m  <- count_buffer(test_m,  farm_sf, 500)

# B4. Bancos en 500 m ---------------------------------------------------------
banco_sf            <- get_osm_points("amenity", "bank")
train$n_bancos_500m <- count_buffer(train_m, banco_sf, 500)
test$n_bancos_500m  <- count_buffer(test_m,  banco_sf, 500)

# B5. Gimnasios en 500 m ------------------------------------------------------
gym_sf           <- get_osm_points("leisure", "fitness_centre")
train$n_gym_500m <- count_buffer(train_m, gym_sf, 500)
test$n_gym_500m  <- count_buffer(test_m,  gym_sf, 500)

# B6. Alumbrado público en 200 m (proxy seguridad nocturna) -------------------
lamp_sf               <- get_osm_points("highway", "street_lamp")
train$n_lamparas_200m <- count_buffer(train_m, lamp_sf, 200)
test$n_lamparas_200m  <- count_buffer(test_m,  lamp_sf, 200)

# =============================================================
# BLOQUE C — BINARIAS CONTEXTUALES
# =============================================================

# C1. Zona de uso residencial -------------------------------------------------
res_poly     <- osmdata_sf(
  opq(bbox = getbb("Bogota Colombia")) |>
    add_osm_feature(key = "landuse", value = "residential")
)$osm_polygons

train$is_residential <- as.integer(lengths(st_within(train, res_poly)) > 0)
test$is_residential  <- as.integer(lengths(st_within(test,  res_poly)) > 0)

# =============================================================
# BLOQUE D — CAPAS ADMINISTRATIVAS (join espacial)
# =============================================================
# D1. Estratos ----------------------------------------------------------------
#     ManzanaEstratificacion.shp — CRS: PCS_CarMAGBOG → transform 4326
estratos <- st_read(here(paths$raw, "ManzanaEstratificacion.shp"), quiet = TRUE) |>
  st_transform(4326) |>
  st_make_valid()

train <- st_join(train, estratos |> dplyr::select(ESTRATO), join = st_intersects)
test  <- st_join(test,  estratos |> dplyr::select(ESTRATO), join = st_intersects)


# D2. UPZ ---------------------------------------------------------------------
#     EPT_UPZ.shp — CRS: PCS_CarMAGBOG → transform 4326
#     EPT = Espacio Público Total (m2/hab) → predictor extra de calidad urbana
upz <- st_read(here(paths$raw, "EPT_UPZ.shp"), quiet = TRUE) |>
  st_transform(4326) |>
  st_make_valid()

train <- st_join(train, upz |> dplyr::select(CODIGO_UPZ, NOMBRE, EPT, AREA_HECTA),
                 join = st_intersects)
test  <- st_join(test,  upz |> dplyr::select(CODIGO_UPZ, NOMBRE, EPT, AREA_HECTA),
                 join = st_intersects)

# D3. Localidades -------------------------------------------------------------
#     Loca.shp — CRS: EPSG:4686 → transform 4326
#     LocCodigo y LocNombre necesarios para CV espacial (04_cv_setup.R)
localidades <- st_read(here(paths$raw, "Loca.shp"), quiet = TRUE) |>
  st_transform(4326) |>
  st_make_valid()

train <- st_join(train, localidades |> dplyr::select(LocCodigo, LocNombre),
                 join = st_intersects)
test  <- st_join(test,  localidades |> dplyr::select(LocCodigo, LocNombre),
                 join = st_intersects)
# =============================================================
# GUARDAR
# =============================================================

saveRDS(train, here(paths$processed, "train_spatial.rds"))
saveRDS(test,  here(paths$processed, "test_spatial.rds"))

message("02_spatial_variables.R   |  train: ",
        nrow(train), " obs  |  test: ", nrow(test), " obs")