
# Función para ahcer prediccion y generar submission
# 02_generar_submission.R

generar_submission <- function(modelo_final, nombre, log_scale = TRUE) {
  tipo <- strsplit(nombre, "_")[[1]][1]
  
  subfolder_map <- list(
    "LR"    = paths$LR,
    "EN"    = paths$EN,
    "CART"  = paths$CART,
    "RF"    = paths$RF,
    "XGB"   = paths$Boosting,
    "LGB"   = paths$Boosting,
    "NNET"  = paths$NN,
    "BRU"   = paths$NN,
    "Super" = paths$Super
  )
  
  subfolder <- subfolder_map[[tipo]]
  if (is.null(subfolder)) stop("Tipo no reconocido: ", tipo)
  
  submission <- predict(modelo_final, test) %>%
    bind_cols(test %>% select(property_id)) %>%
    mutate(.pred = if (log_scale) exp(.pred) else .pred) %>%
    select(property_id, price = .pred)
  
  write_csv(submission, here(subfolder, paste0(nombre, ".csv")))
  message("Submission guardada: ", nombre)
}