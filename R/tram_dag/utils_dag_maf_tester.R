library(keras)
library(tensorflow)
source('R/tram_dag/utils_dag_maf.R')

#Defines MAF
hidden_features = c(4,4)
adjacency <- matrix(c(0, 1, 1, 0, 0, 0, 0, 0, 0), nrow = 3, byrow = FALSE)

# Create Masks
masks = create_masks(adjacency = adjacency, hidden_features)
layer_sizes <- c(ncol(adjacency), hidden_features, nrow(adjacency))

# Plot MAF
dag_maf_plot(masks, layer_sizes)

# Create MAF
input_layer <- layer_input(shape = list(ncol(adjacency)))
d = input_layer
for (i in 2:(length(layer_sizes) - 1)) {
  d = LinearMasked(units=layer_sizes[i], mask=t(masks[[i-1]]))(d)
  d = layer_activation(activation='relu')(d)
}
out = LinearMasked(units=layer_sizes[length(layer_sizes)], mask=t(masks[[length(layer_sizes) - 1]]))(d)
model = keras_model(inputs = input_layer, outputs = out)

model
x = tf$ones(c(1L,3L))
model(x)

# Calculate the Jacobian matrix using Python and TensorFlow
with(tf$GradientTape(persistent = TRUE) %as% tape, {
  tape$watch(x)
  y <- model(x)
})
tape$jacobian(y, x)
  
  
if (FALSE){
  library(reticulate)
  source_python("R/tram_dag/dag_maf.py")
  hidden_features_r = hidden_features
  adjacency_np <- reticulate::r_to_py(adjacency)
  hidden_features_np <- reticulate::r_to_py(hidden_features_r)
  masks_np <- create_masks_np(adjacency_np, hidden_features_np)
  
  for (i in 1:length(masks)) {
    if (sum(masks_np[[i]] != masks[[i]]) > 0){
      print("Error")
    } else {
      print("OK")
    }
  }
}







