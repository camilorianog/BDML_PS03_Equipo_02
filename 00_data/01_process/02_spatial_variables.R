# ============================================================
# 02_spatial_variables.R
# Variables espaciales derivadas de OpenStreetMap
# ============================================================
# Team 02 — Problem Set 03
# ============================================================

nosleep_on(keep_display = FALSE)

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

#' Helper: llama osmdata_sf rotando servidores Overpass en caso de 429/403
osm_fetch_with_retry <- function(q, max_tries = 8, wait_sec = 15) {
  
  # Se elimina maps.mail.ru porque suele devolver 403 en redes institucionales
  overpass_servers <- c(
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter"
  )
  
  for (i in seq_len(max_tries)) {
    
    server <- overpass_servers[((i - 1) %% length(overpass_servers)) + 1]
    osmdata::set_overpass_url(server)
    
    # Espera exponencial: 15s, 15s, 30s, 30s, 45s, 45s, ...
    wait_actual <- wait_sec * ceiling(i / 2)
    
    result <- tryCatch(
      osmdata_sf(q),
      error = function(e) {
        msg <- conditionMessage(e)
        # Captura 429 (rate limit) Y 403 (forbidden / bloqueado)
        if (grepl("429|403|Too Many|Forbidden|rate.limit|backoff",
                  msg, ignore.case = TRUE)) {
          message(
            "  [", sub(".*HTTP (\\d+).*", "HTTP \\1", msg), "]",
            " en ", server,
            " — cambiando servidor, espero ", wait_actual, "s",
            " (intento ", i, "/", max_tries, ")"
          )
          Sys.sleep(wait_actual)
          NULL
        } else {
          stop(e)   # otros errores sí se propagan
        }
      }
    )
    if (!is.null(result)) return(result)
  }
  stop("Todos los servidores Overpass fallaron tras ", max_tries, " intentos.")
}

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
  
  raw <- osm_fetch_with_retry(q)
  
  pts <- raw$osm_points
  if (is.null(pts) || nrow(pts) == 0) {
    # fallback: centroides de polígonos (e.g. cafés mapeados como edificio)
    polys <- raw$osm_polygons
    pts <- if (!is.null(polys) && nrow(polys) > 0) suppressWarnings(st_centroid(polys)) else NULL
  }
  if (is.null(pts) || nrow(pts) == 0) {
    warning("  Sin geometría encontrada para: ", key, " = ", value)
    return(sf::st_sf(osm_id = character(0), geometry = sf::st_sfc(crs = 4326)))
  }
  obj <- pts |>
    dplyr::select(osm_id) |>
    st_as_sf(crs = 4326)
  
  saveRDS(obj, cache_file)
  
  obj
}

#' Descarga polígonos OSM para key = value en bbox de Bogotá
get_osm_polygons <- function(key, value, bbox = bbox_bogota) {
  
  cache_file <- here(
    osm_cache,
    paste0(key, "_", value, "_poly.rds")
  )
  
  if (file.exists(cache_file)) {
    message("  Loading cache: ", key, " = ", value, " (polygons)")
    return(readRDS(cache_file))
  }
  
  message("  OSM polygons: ", key, " = ", value)
  
  q <- opq(bbox = bbox) |>
    add_osm_feature(key = key, value = value)
  
  raw <- osm_fetch_with_retry(q)
  
  # Combinar polygons y multipolygons (en OSM los parques suelen ser relaciones)
  polys  <- raw$osm_polygons
  mpolys <- raw$osm_multipolygons
  
  combined <- dplyr::bind_rows(
    if (!is.null(polys)  && nrow(polys)  > 0) dplyr::select(polys,  osm_id) else NULL,
    if (!is.null(mpolys) && nrow(mpolys) > 0) dplyr::select(mpolys, osm_id) else NULL
  )
  
  if (is.null(combined) || nrow(combined) == 0) {
    warning("  Sin polígonos encontrados para: ", key, " = ", value)
    return(sf::st_sf(osm_id = character(0), geometry = sf::st_sfc(crs = 4326)))
  }
  
  obj <- combined |> st_make_valid()
  
  saveRDS(obj, cache_file)
  
  obj
}

#' sf vacío con geometría POINT (misma clase que st_centroid / get_osm_points)
empty_point_sf <- function() {
  st_sf(osm_id = "._", geometry = st_sfc(st_point(c(0, 0)), crs = 4326))[0, ]
}

#' Distancia mínima (metros) desde cada fila de base_sf a target_sf
#' versión eficiente
dist_min <- function(base_sf, target_sf) {
  if (nrow(target_sf) == 0L) {
    return(rep(NA_real_, nrow(base_sf)))
  }
  
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
parques_poly <- get_osm_polygons("leisure", "park")
parques_sf <- if (nrow(parques_poly) > 0L) {
  suppressWarnings(st_centroid(parques_poly))
} else {
  empty_point_sf()
}
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
res_poly             <- get_osm_polygons("landuse", "residential")
train$is_residential <- as.integer(lengths(st_within(train, res_poly)) > 0)
test$is_residential  <- as.integer(lengths(st_within(test,  res_poly)) > 0)

# =============================================================
# BLOQUE D — CAPAS ADMINISTRATIVAS (join espacial)
# =============================================================

assets_d <- here("00_data", "01_process", "02a_spatial_variables_assets")

# D1. Estratos ----------------------------------------------------------------
estratos <- st_read(
  here(assets_d, "01_estratos", "ManzanaEstratificacion.shp"),
  quiet = TRUE
) |>
  st_transform(4326) |>
  st_make_valid()

train <- st_join(train, estratos |> dplyr::select(ESTRATO), join = st_intersects)
test  <- st_join(test,  estratos |> dplyr::select(ESTRATO), join = st_intersects)

# D2. UPZ ---------------------------------------------------------------------
upz <- st_read(
  here(assets_d, "02_UPZ", "EPT_UPZ.shp"),
  quiet = TRUE
) |>
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
localidades <- st_read(
  here(assets_d, "03_localidades", "Loca.shp"),
  quiet = TRUE
) |>
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

message("02_spatial_variables.R   |  train: ",
        nrow(train), " obs  |  test: ", nrow(test), " obs")

# --- Limpieza de objetos intermedios ----------------------------------------

rm(
  bbox_bogota, osm_cache,
  osm_fetch_with_retry, get_osm_points, get_osm_polygons, empty_point_sf,
  dist_min, count_buffer,
  cafes_sf, bus_sf, metro_sf, hospital_sf, colegio_sf, univ_sf,
  parques_poly, parques_sf, super_sf, rest_sf, farm_sf, banco_sf, gym_sf, lamp_sf,
  res_poly, estratos, upz, localidades,
  assets_d, train_m, test_m
)
gc()

nosleep_off()