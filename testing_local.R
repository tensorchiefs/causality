remove.packages("deepregression")
install.packages("/Users/oli/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/Causality_2022/workspace/deepregression/", repos = NULL, type = "source")
install.packages("/Users/oli/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/Causality_2022/workspace/deeptrafo/", repos = NULL, type = "source")
library(deepregression)
library(deeptrafo)

# Prepare the data
data("wine", package = "ordinal")
wine$z <- rnorm(nrow(wine))
wine$x <- rnorm(nrow(wine))

# Set up neural network architecture
nn <- \(x) x |>
  layer_dense(input_shape = 1L, units = 2L, activation = "relu") |>
  layer_dense(1L)

# Model formula and definition
fml <- rating ~ 0 + temp + contact + s(z, df = 3) + nn(x)
m <- deeptrafo(fml, wine, latent_distr = "logistic", monitor_metric = NULL,
               return_data = TRUE, list_of_deep_models = list(nn = nn))

for (l in m$model$layers){
  print(l$name)
}

m$model$layers
# Overview
print(m)
