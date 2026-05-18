
# Función para guardar detalles de modelo en un log general
# 01_log_modelo.R

log_modelo <- function(tuned, best, nombre) {
  tipo      <- strsplit(nombre, "_")[[1]][1]
  cv_mae    <- show_best(tuned, metric = "mae", n = 1) %>% pull(mean)
  autor     <- Sys.info()[["user"]]
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  
  entry <- tibble(
    timestamp = timestamp,
    nombre    = nombre,
    tipo      = tipo,
    cv_mae    = round(cv_mae, 0),
    autor     = autor
  )
  
  log_path <- here(paths$submissions, "submission_log.csv")
  
  if (file.exists(log_path)) {
    write_csv(entry, log_path, append = TRUE)
  } else {
    write_csv(entry, log_path)
  }
  
  message("Log actualizado: ", nombre, " | MAE: ", round(cv_mae, 0))
}