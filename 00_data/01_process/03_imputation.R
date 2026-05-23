# ============================================================
# 03_imputation.R
# Imputación de missings con flags _was_na
# ============================================================
# Patrones de missing observados:
#   train → ESTRATO: 13,167 NAs | CODIGO_UPZ: 6 NAs  | LocCodigo: 0
#   test  → ESTRATO:  3,172 NAs | CODIGO_UPZ: 49 NAs | LocCodigo: 0
#
#   ESTRATO: NAs en zonas no residenciales (comercial/industrial)
#            → imputar por moda dentro de CODIGO_UPZ × property_type
#   CODIGO_UPZ: propiedades en borde de Bogotá fuera de polígonos
#            → imputar por moda dentro de LocCodigo × property_type
#   surface_total/covered: altísimos NAs (82% en test)
#            → imputar por media dentro de CODIGO_UPZ × property_type
#
# Estrategia general:
#   Numéricas   → media condicional por grupo → fallback media global
#   Categóricas → moda condicional por grupo  → fallback moda global
#   Siempre se genera flag _was_na como predictor adicional
# ============================================================

# --- Funciones ---------------------------------------------------------------

get_mode <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_character_)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

impute_numeric <- function(data, vars, groups) {
  data <- as.data.frame(data)
  key  <- interaction(data[groups], drop = TRUE, lex.order = TRUE)
  
  for (v in intersect(vars, names(data))) {
    # Flag antes de imputar
    data[[paste0(v, "_was_na")]] <- as.integer(is.na(data[[v]]))
    
    gm       <- tapply(data[[v]], key, \(z) mean(z, na.rm = TRUE))
    gm_vec   <- gm[as.character(key)]
    global_m <- mean(data[[v]], na.rm = TRUE)
    gm_vec[is.na(gm_vec)] <- global_m
    
    na_idx            <- is.na(data[[v]])
    data[[v]][na_idx] <- gm_vec[na_idx]
  }
  data
}

impute_categorical <- function(data, vars, groups) {
  data <- as.data.frame(data)
  key  <- interaction(data[groups], drop = TRUE, lex.order = TRUE)
  
  for (v in intersect(vars, names(data))) {
    # Flag antes de imputar
    data[[paste0(v, "_was_na")]] <- as.integer(is.na(data[[v]]))
    
    x_chr       <- as.character(data[[v]])
    global_mode <- get_mode(x_chr)
    gm          <- tapply(x_chr, key, get_mode)
    gm_vec      <- gm[as.character(key)]
    gm_vec[is.na(gm_vec)] <- global_mode
    
    na_idx        <- is.na(x_chr)
    x_chr[na_idx] <- gm_vec[na_idx]
    
    data[[v]] <- if (is.factor(data[[v]])) factor(x_chr) else x_chr
  }
  data
}

# --- Cargar ------------------------------------------------------------------

as_modeling_df <- function(x) {
  if (inherits(x, "sf")) {
    x <- sf::st_drop_geometry(x)
  }
  as.data.frame(x)
}

if (exists("train", envir = .GlobalEnv)) {
  train <- as_modeling_df(get("train", envir = .GlobalEnv))
  message("03_imputation.R  |  usando train en memoria")
} else if (file.exists(here(paths$processed, "train_model.rds"))) {
  train <- readRDS(here(paths$processed, "train_model.rds")) |> as.data.frame()
  message("03_imputation.R  |  cargando train_model.rds desde processed")
} else {
  stop(
    "No hay 'train' en memoria ni train_model.rds en processed. Ejecuta el pipeline desde 00_clean.R.",
    call. = FALSE
  )
}

if (exists("test", envir = .GlobalEnv)) {
  test <- as_modeling_df(get("test", envir = .GlobalEnv))
  message("03_imputation.R  |  usando test en memoria")
} else if (file.exists(here(paths$processed, "test_model.rds"))) {
  test <- readRDS(here(paths$processed, "test_model.rds")) |> as.data.frame()
  message("03_imputation.R  |  cargando test_model.rds desde processed")
} else {
  stop(
    "No hay 'test' en memoria ni test_model.rds en processed. Ejecuta el pipeline desde 00_clean.R.",
    call. = FALSE
  )
}

# --- PASO 1: CODIGO_UPZ primero (lo necesitamos para imputar el resto) -------
# Solo 6 en train y 49 en test → imputar por LocCodigo × property_type

train <- impute_categorical(train, "CODIGO_UPZ", c("LocCodigo", "property_type"))
test  <- impute_categorical(test,  "CODIGO_UPZ", c("LocCodigo", "property_type"))

# --- PASO 2: ESTRATO ---------------------------------------------------------
# 13k NAs en train (zonas no residenciales)
# Imputar por CODIGO_UPZ × property_type

train <- impute_categorical(train, "ESTRATO", c("CODIGO_UPZ", "property_type"))
test  <- impute_categorical(test,  "ESTRATO",  c("CODIGO_UPZ", "property_type"))

# --- PASO 3: Variables numéricas estructurales -------------------------------
# surface_total tiene 82% NAs en test → crítico imputar bien

vars_num <- c(
  "surface_total", "surface_covered",
  "rooms", "bathrooms"
  # bedrooms NO: tiene 0% NAs
)

train <- impute_numeric(train, vars_num, c("CODIGO_UPZ", "property_type"))
test  <- impute_numeric(test,  vars_num, c("CODIGO_UPZ", "property_type"))

# --- PASO 4: Factorizar ------------------------------------------------------

train <- train |> mutate(
  property_type = as.factor(property_type),
  ESTRATO       = as.factor(ESTRATO),
  CODIGO_UPZ    = as.factor(CODIGO_UPZ),
  LocCodigo     = as.factor(LocCodigo)
)

test <- test |> mutate(
  property_type = as.factor(property_type),
  ESTRATO       = as.factor(ESTRATO),
  CODIGO_UPZ    = as.factor(CODIGO_UPZ),
  LocCodigo     = as.factor(LocCodigo)
)

# --- PASO 5: Eliminar price == NA en train -----------------------------------

n_antes <- nrow(train)
train   <- train |> drop_na(price)
message("  Obs removidas por price == NA: ", n_antes - nrow(train))

# --- PASO 6: Imputar variables de texto (NAs por descripciones vacías) -------

vars_texto <- c(
  "parqueadero", "parqueadero_inv", "piscina", "piscina_privada",
  "ascensor", "sin_ascensor", "terraza", "zona_exterior_priv",
  "salon_comunal", "sala_reuniones", "porteria", "vigilancia",
  "conjunto", "remodelada", "antiguo", "moderno",
  "cocina_integral", "cocina_americana", "gym", "chimenea",
  "pisos_madera", "walk_in_closet", "lavanderia", "duplex",
  "apartaestudio", "vista", "vista_cerros",
  "tenis", "golf", "squash", "estudio_cuarto", "escaleras_int"
)

vars_texto <- intersect(vars_texto, names(train))

train <- train |>
  mutate(across(all_of(vars_texto), \(x) replace_na(x, 0)))
test  <- test  |>
  mutate(across(all_of(vars_texto), \(x) replace_na(x, 0)))

# --- PASO 7: NOMBRE/EPT/AREA_HECTA (fuera de polígono UPZ en train y test) --

train <- train |>
  mutate(
    NOMBRE     = if_else(is.na(NOMBRE),     get_mode(NOMBRE),              NOMBRE),
    EPT        = if_else(is.na(EPT),        mean(EPT,        na.rm = TRUE), EPT),
    AREA_HECTA = if_else(is.na(AREA_HECTA), mean(AREA_HECTA, na.rm = TRUE), AREA_HECTA)
  )

test <- test |>
  mutate(
    NOMBRE     = if_else(is.na(NOMBRE),     get_mode(NOMBRE),              NOMBRE),
    EPT        = if_else(is.na(EPT),        mean(EPT,        na.rm = TRUE), EPT),
    AREA_HECTA = if_else(is.na(AREA_HECTA), mean(AREA_HECTA, na.rm = TRUE), AREA_HECTA)
  )
# --- Verificación final ------------------------------------------------------

missings_train <- train |>
  summarise(across(everything(), \(x) sum(is.na(x)))) |>
  pivot_longer(everything(), names_to = "var", values_to = "n_na") |>
  filter(n_na > 0) |>
  arrange(desc(n_na))

missings_test <- test |>
  summarise(across(everything(), \(x) sum(is.na(x)))) |>
  pivot_longer(everything(), names_to = "var", values_to = "n_na") |>
  filter(n_na > 0) |>
  arrange(desc(n_na))

message("  Variables con NAs restantes en train: ", nrow(missings_train))
message("  Variables con NAs restantes en test:  ", nrow(missings_test))

if (nrow(missings_train) > 0) print(missings_train)
if (nrow(missings_test)  > 0) print(missings_test)


# --- Guardar -----------------------------------------------------------------

saveRDS(train, here(paths$processed, "train_model.rds"))
saveRDS(test,  here(paths$processed, "test_model.rds"))

message("03_imputation.R  |  train: ",
        nrow(train), " obs  |  test: ", nrow(test), " obs")

# --- Limpieza de objetos intermedios ----------------------------------------

rm(get_mode, impute_numeric, impute_categorical, as_modeling_df,
   vars_num, vars_texto, n_antes, missings_train, missings_test)
gc()