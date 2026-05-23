
# Función para guardar detalles de modelo en un log general
# 01_log_modelo.R

log_modelo <- function(tuned = NULL, nombre, best = NULL, mae_directo = NULL) {
  tipo      <- strsplit(nombre, "_")[[1]][1]

  # mae_directo → valor numérico explícito (p.ej. SuperLearner, ranger directo)
  # fit_resamples → collect_metrics | tune_grid/tune_bayes → show_best
  if (!is.null(mae_directo)) {
    cv_mae <- mae_directo
  } else if (inherits(tuned, "tune_results")) {
    cv_mae <- show_best(tuned, metric = "mae", n = 1) %>% pull(mean)
  } else {
    cv_mae <- collect_metrics(tuned) %>% filter(.metric == "mae") %>% pull(mean)
  }

  autor     <- Sys.info()[["user"]]
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  
  entry <- tibble(
    timestamp = timestamp,
    nombre    = nombre,
    tipo      = tipo,
    cv_mae    = round(cv_mae, 4),
    autor     = autor
  )
  
  log_path <- here(paths$submissions, "submission_log.csv")
  
  if (file.exists(log_path)) {
    write_csv(entry, log_path, append = TRUE)
  } else {
    write_csv(entry, log_path)
  }
  
  message("Log actualizado: ", nombre, " | MAE: ", round(cv_mae, 4))
}