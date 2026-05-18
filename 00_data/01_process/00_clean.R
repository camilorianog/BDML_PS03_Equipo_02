# ============================================================
# 00_clean.R
# Limpieza general del dataset
# ============================================================

# --- Cargar datos -------------------------------------------

train <- read.csv(here(paths$competition, "train.csv"))
test  <- read.csv(here(paths$competition, "test.csv"))

# --- Variables a excluir ------------------------------------

excluir <- c(
  "city", "operation_type"
)

# --- Aplicar exclusiones ------------------------------------

train <- train |> select(-any_of(excluir))
test  <- test  |> select(-any_of(excluir))

# --- Verificar stats descriptivas --------------------------------------------

skim(train |> select(-property_id))

# --- Winsorización de outliers extremos ---------------------

winsorizr <- function(x, p = 0.99) {
  cap_upper <- quantile(x, p, na.rm = TRUE)
  cap_lower <- quantile(x, 1 - p, na.rm = TRUE)
  pmin(pmax(x, cap_lower), cap_upper)
}

vars_winsorizar <- c(
  "surface_total",
  "surface_covered"
)

train <- train |> mutate(across(all_of(vars_winsorizar), winsorizr))
test  <- test  |> mutate(across(all_of(vars_winsorizar), winsorizr))

