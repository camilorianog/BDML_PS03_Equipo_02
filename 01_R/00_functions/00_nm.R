
# Función para generar nombre de modelo para modelos basados en tiny models
# 00_nm.R


nm <- function(tipo, best) {
  params <- best %>% 
    select(-any_of(c(".metric", ".config", ".estimator", "n", "std_err", "mean")))
  
  param_str <- paste(
    names(params),
    sapply(params, function(x) gsub("\\.", "", as.character(x))),
    sep = "", collapse = "_"
  )
  
  paste(tipo, param_str, sep = "_")
}