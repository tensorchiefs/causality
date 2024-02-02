library(keras)
library(tensorflow)
source('R/tram_dag/utils_dag_maf.R')

#Defines MAF
hidden_features = c(2,2)
adjacency <- matrix(c(0, 1, 1, 0, 0, 0, 0, 0, 0), nrow = 3, byrow = FALSE)

# Create Masks
masks = create_masks(adjacency = adjacency, hidden_features)
layer_sizes <- c(ncol(adjacency), hidden_features, nrow(adjacency))

# Plot MAF
dag_maf_plot(masks, layer_sizes)

# masks_expansion = function(masks, order=2){
#   masks_out = list()
#   for (i in 1:length(masks)){
#     mm = masks[[i]]
#     for (r in (1:(order-1))) {
#       mm = rbind(mm,masks[[i]])
#     }
#     masks_out[[i]] = mm
#   } 
#   return (masks_out)
# } 
# masks = masks_expansion(masks, 2)
#layer_sizes <- c(ncol(adjacency), 2*hidden_features, 2*nrow(adjacency))
dag_maf_plot(masks, layer_sizes)

# Create MAF
model = create_theta_tilde_maf(adjacency = adjacency, order = 10L)
x = tf$ones(c(1L,3L))
r = model(x)
r


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







