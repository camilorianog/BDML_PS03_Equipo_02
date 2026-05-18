# ============================================================
# 01_text_variables.R
# Variables derivadas de la descripción del inmueble
# ============================================================

options(max.print = 25)

# --- Tokenización -----------------------------------------------------------

descripcion_tokenizada <- tokenize_words(train$description)

# --- Eliminar stopwords -----------------------------------------------------------

lista_palabras1 <- stopwords(language = "es", source = "snowball")
lista_palabras2 <- stopwords(language = "es", source = "nltk")
lista_palabras <- union(lista_palabras1, lista_palabras2)
lista_palabras

descripcion_tokenizada <- lapply(descripcion_tokenizada, function(tokens) {
  setdiff(tokens, lista_palabras)
})

# Tocaba hacerle lapply ya que la variable son muchos textos, no un solo texto como en el html del profe

# --- Stemming ----------------------------------------------------------------

#descripcion_tokenizada <- lapply(descripcion_tokenizada, function(tokens) {
#  wordStem(tokens, "spanish")
#})


# --- Frecuencia de palabras --------------------------------------------------

frecuencia <- unlist(descripcion_tokenizada) %>%
  table() %>%
  data.frame() %>%
  rename("Palabra" = ".") %>%
  arrange(desc(Freq))

# rm(descripcion_tokenizada,frecuencia,excluir,lista_palabras,lista_palabras1,lista_palabras2)

# --- Construcción de regresores ----------------------------------------------

text_variables <- function(db) {
  db %>%
    mutate(
      # --- Parqueadero ---
      parqueadero        = as.integer(str_detect(description,
                                                 "parqueadero|parqueaderos|garaje|garajes|garage|garages")),
      parqueadero_inv    = as.integer(str_detect(description,
                                                 "parqueadero.{0,15}invitad|parqueadero.{0,15}visit")),
      
      # --- Piscina / spa ---
      piscina            = as.integer(str_detect(description,
                                                 "piscina|piscinas|jacuzzi|jacuzzis|turco|turcos|sauna|saunas")),
      piscina_privada    = as.integer(str_detect(description,
                                                 "piscina privada|piscinas privadas|piscina propia|piscina exclusiva")),
      
      # --- Ascensor ---
      ascensor           = as.integer(str_detect(description, regex(
        "(?<!sin )ascensor|(?<!sin )ascensores|(?<!sin )elevador|(?<!sin )elevadores"))),
      sin_ascensor       = as.integer(str_detect(description,
                                                 "sin ascensor|sin ascensores|no tiene ascensor|no hay ascensor|sin elevador")),
      
      # --- Terraza / balcon (balcón → balco?n | balcn) ---
      terraza            = as.integer(
        str_detect(description, "terraza|terrazas|balco?n|balcones") &
          !str_detect(description, "terraza privada|terrazas privadas|terraza propia")
      ),
      
      # --- Zona exterior privada (jardín → jardi?n | jardn) ---
      zona_exterior_priv = as.integer(str_detect(description,
                                                 "jardi?n privado|jardines privados|jardi?n propio|jardines propios|patio privado|patios privados|patio propio|patios propios|terraza privada|terrazas privadas|terraza propia|a?rea exterior privada")),
      
      # --- Espacios comunes (salón → salo?n | saln; reunión → reunio?n | reunin) ---
      salon_comunal      = as.integer(str_detect(description,
                                                 "zona social|salo?n social|salones sociales|salo?n comunal|salones comunales")),
      sala_reuniones     = as.integer(str_detect(description,
                                                 "sala de reunio?n|salo?n de reunio?n|salas de reuniones|sala de juntas|salo?n de juntas|salas de juntas")),
      
      # --- Portería (portería → porteri?a | portera) ---
      porteria           = as.integer(str_detect(description,
                                                 "porteri?a|porteri?as|lobby|lobbies|recepcio?n|sala de espera")),
      
      # --- Seguridad (cámaras → ca?maras | cmaras) ---
      vigilancia         = as.integer(str_detect(description,
                                                 "vigilancia|ca?maras|cctv|seguridad")),
      
      # --- Conjunto (urbanización → urbanizacio?n | urbanizacin) ---
      conjunto           = as.integer(str_detect(description,
                                                 "conjunto cerrado|conjunto privado|conjunto residencial|unidad residencial|urbanizacio?n|urbanizaciones")),
      
      # --- Estado ---
      # remodelación → remodelacio?n | remodelacin
      # renovación  → renovacio?n   | renovacin
      remodelada         = as.integer(str_detect(description,
                                                 "remodelado|remodelada|remodelados|remodeladas|remodelacio?n|renovado|renovada|renovados|renovadas|renovacio?n")),
      
      # clásico → cla?sico | clsico
      antiguo            = as.integer(str_detect(description,
                                                 "\\bantiguo\\b|\\bantigua\\b|\\bantiguos\\b|\\bantiguas\\b|edificio antiguo|casa antigua|estilo cla?sico|para remodelar|a remodelar|necesita remodelar|oportunidad de negocio")),
      
      # contemporáneo → contempora?neo | contemporneo
      # diseño → dise[nñ]o (ñ puede mantenerse o convertirse a n)
      moderno            = as.integer(str_detect(description,
                                                 "moderno|moderna|modernos|modernas|contempora?neo|contempora?nea|contempora?neos|contempora?neas|acabados modernos|dise[n\u00f1]o moderno|dise[n\u00f1]os modernos")),
      
      # --- Cocinas ---
      cocina_integral    = as.integer(str_detect(description,
                                                 "cocina integral|cocinas integrales")),
      cocina_americana   = as.integer(str_detect(description,
                                                 "cocina americana|cocinas americanas|cocina abierta|cocinas abiertas|tipo americano")),
      
      # --- Deporte y recreacion ---
      gym                = as.integer(str_detect(description,
                                                 "gimnasio|gimnasios|gym|fitness")),
      tenis              = as.integer(str_detect(description,
                                                 "tenis|cancha de tenis|canchas de tenis|court de tenis|courts de tenis")),
      golf               = as.integer(str_detect(description,
                                                 "golf|cancha de golf|canchas de golf|campo de golf|campos de golf")),
      squash             = as.integer(str_detect(description,
                                                 "squash|cancha de squash|canchas de squash")),
      
      # --- Caracteristicas internas ---
      chimenea           = as.integer(str_detect(description,
                                                 "chimenea|chimeneas")),
      pisos_madera       = as.integer(str_detect(description,
                                                 "piso de madera|pisos de madera|piso en madera|pisos en madera|laminado|laminados|parquet")),
      walk_in_closet     = as.integer(str_detect(description,
                                                 "walk.?in.?closet|walking.?closet|walkin.?closet|walk in|vestier|vestiers")),
      
      # lavandería → lavanderi?a | lavandera
      lavanderia         = as.integer(str_detect(description,
                                                 "lavanderi?a|lavanderi?as|zona de lavado|zonas de lavado|cuarto de servicio|cuartos de servicio|cuarto de empleada|cuartos de empleada|cuarto de ropas|cuarto ropas")),
      
      # habitación → habitacio?n | habitacin
      estudio_cuarto     = as.integer(str_detect(description,
                                                 "\\bestudio\\b|\\bestudios\\b|biblioteca|bibliotecas|cuarto de estudio|cuartos de estudio|habitacio?n de estudio")),
      
      # dúplex → du?plex | dplex
      duplex             = as.integer(str_detect(description, "du?plex")),
      
      escaleras_int      = as.integer(str_detect(description,
                                                 "escalera interna|escaleras internas|escalera interior|escaleras interiores|escalera propia|escaleras propias|escalera privada|escaleras privadas")),
      
      # --- Tipo de inmueble ---
      apartaestudio      = as.integer(str_detect(description,
                                                 "apartaestudio|apartaestudios|apto estudio|apartamento estudio|apartamentos estudio")),
      
      # --- Vistas (panorámica → panora?mica | panormica) ---
      vista              = as.integer(str_detect(description,
                                                 "vista exterior|vista panora?mica|vista panora?micas|vista ciudad|vista a la ciudad|vista al ciudad|vista abierta|hermosa vista|bella vista|linda vista")),
      vista_cerros       = as.integer(str_detect(description,
                                                 "vista cerros|vista a los cerros|vista al cerro|cerros orientales|cerros orient")),
    )
}

train <- text_variables(train)
test  <- text_variables(test)
