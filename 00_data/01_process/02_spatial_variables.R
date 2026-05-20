# ============================================================
# 02_spatial_variables.R
# Variables espaciales derivadas de OpenStreetMap
# ============================================================
# Team 02 — Problem Set 03
# ============================================================

# --- Helpers -----------------------------------------------------------------

options(timeout = 2500)

# bbox fija basada en cobertura real de los datos
bbox_bogota <- c(
  min_lon = -74.3,
  min_lat =  4.6,
  max_lon = -74,
  max_lat =  4.9
)

# cache local OSM
osm_cache <- here(paths$raw, "osm_cache")
dir.create(osm_cache, recursive = TRUE, showWarnings = FALSE)

#' Descarga puntos OSM para key = value en bbox de Bogotá
get_osm_points <- function(key, value, bbox = bbox_bogota) {
  
  cache_file <- here(
    osm_cache,
    paste0(key, "_", value, ".rds")
  )
  
  # usar cache si existe
  if (file.exists(cache_file)) {
    message("  Loading cache: ", key, " = ", value)
    return(readRDS(cache_file))
  }
  
  message("  OSM: ", key, " = ", value)
  
  q <- opq(bbox = bbox) |>
    add_osm_feature(key = key, value = value)
  
  raw <- osmdata_sf(q)
  
  obj <- raw$osm_points |>
    dplyr::select(osm_id) |>
    st_as_sf(crs = 4326)
  
  saveRDS(obj, cache_file)
  
  obj
}

#' Distancia mínima (metros) desde cada fila de base_sf a target_sf
#' versión eficiente
dist_min <- function(base_sf, target_sf) {
  
  idx <- st_nearest_feature(base_sf, target_sf)
  
  nearest <- target_sf[idx, ]
  
  as.numeric(
    st_distance(
      base_sf,
      nearest,
      by_element = TRUE
    )
  )
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
univ_sf                <- get_osm_points("amenity", "university")
train$dist_universidad <- dist_min(train, univ_sf)
test$dist_universidad  <- dist_min(test,  univ_sf)

# A7. Parques -----------------------------------------------------------------
q_parques   <- opq(bbox = bbox_bogota) |>
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

# B6. Alumbrado público en 200 m ----------------------------------------------
lamp_sf               <- get_osm_points("highway", "street_lamp")
train$n_lamparas_200m <- count_buffer(train_m, lamp_sf, 200)
test$n_lamparas_200m  <- count_buffer(test_m,  lamp_sf, 200)

# =============================================================
# BLOQUE C — BINARIAS CONTEXTUALES
# =============================================================

# C1. Zona de uso residencial -------------------------------------------------
res_poly <- osmdata_sf(
  opq(bbox = bbox_bogota) |>
    add_osm_feature(key = "landuse", value = "residential")
)$osm_polygons

train$is_residential <- as.integer(lengths(st_within(train, res_poly)) > 0)
test$is_residential  <- as.integer(lengths(st_within(test,  res_poly)) > 0)

# =============================================================
# BLOQUE D — CAPAS ADMINISTRATIVAS (join espacial)
# =============================================================

# D1. Estratos ----------------------------------------------------------------
estratos <- st_read(here(paths$raw, "ManzanaEstratificacion.shp"), quiet = TRUE) |>
  st_transform(4326) |>
  st_make_valid()

train <- st_join(train, estratos |> dplyr::select(ESTRATO), join = st_intersects)
test  <- st_join(test,  estratos |> dplyr::select(ESTRATO), join = st_intersects)

# D2. UPZ ---------------------------------------------------------------------
upz <- st_read(here(paths$raw, "EPT_UPZ.shp"), quiet = TRUE) |>
  st_transform(4326) |>
  st_make_valid()

train <- st_join(
  train,
  upz |> dplyr::select(CODIGO_UPZ, NOMBRE, EPT, AREA_HECTA),
  join = st_intersects
)

test <- st_join(
  test,
  upz |> dplyr::select(CODIGO_UPZ, NOMBRE, EPT, AREA_HECTA),
  join = st_intersects
)

# D3. Localidades -------------------------------------------------------------
localidades <- st_read(here(paths$raw, "Loca.shp"), quiet = TRUE) |>
  st_transform(4326) |>
  st_make_valid()

train <- st_join(
  train,
  localidades |> dplyr::select(LocCodigo, LocNombre),
  join = st_intersects
)

test <- st_join(
  test,
  localidades |> dplyr::select(LocCodigo, LocNombre),
  join = st_intersects
)

# =============================================================
# GUARDAR
# =============================================================

saveRDS(train, here(paths$processed, "train_spatial.rds"))
saveRDS(test,  here(paths$processed, "test_spatial.rds"))

message("02_spatial_variables.R   |  train: ",
        nrow(train), " obs  |  test: ", nrow(test), " obs")