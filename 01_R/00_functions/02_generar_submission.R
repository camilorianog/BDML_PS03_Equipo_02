# ============================================================
# 02_generar_submission.R
# Genera el archivo CSV de submission para Kaggle
# ============================================================


generar_submission <- function(modelo_final, nombre) {
  
  # Identificar subcarpeta según prefijo del nombre
  tipo <- strsplit(nombre, "_")[[1]][1]
  
  subfolder_map <- list(
    "LR"    = paths$LR,
    "EN"    = paths$EN,
    "CART"  = paths$CART,
    "RF"    = paths$RF,
    "XGB"   = paths$Boosting,
    "LGB"   = paths$Boosting,
    "NNET"  = paths$NN,
    "Super" = paths$Super
  )
  
  subfolder <- subfolder_map[[tipo]]
  if (is.null(subfolder)) stop("Tipo no reconocido: ", tipo,
                               ". Usa LR, EN, CART, RF, XGB, LGB, NNET o Super.")
  
  # Predecir — tidymodels devuelve tibble con columna .pred
  # Los modelos predicen en log(price) → exp() para volver a pesos
  submission <- predict(modelo_final, new_data = test) |>
    mutate(price = exp(.pred)) |>
    bind_cols(test |> select(property_id)) |>
    select(property_id, price)
  
  # Verificar estructura
  stopifnot(
    nrow(submission) == nrow(test),
    all(c("property_id", "price") %in% names(submission)),
    !any(is.na(submission$price))
  )
  
  # Guardar
  write_csv(submission, here(subfolder, paste0(nombre, ".csv")))
  
  message(
    "Submission guardada: ", nombre, ".csv",
    "  |  n: ", nrow(submission),
    "  |  mean: $", format(round(mean(submission$price), 0), big.mark = ","),
    "  |  min: $",  format(round(min(submission$price),  0), big.mark = ","),
    "  |  max: $",  format(round(max(submission$price),  0), big.mark = ",")
  )
}