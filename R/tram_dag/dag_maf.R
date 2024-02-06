create_masks <- function(adjacency, hidden_features=c(64, 64), activation='relu') {
  out_features <- nrow(adjacency)
  in_features <- ncol(adjacency)
  
  #adjacency_unique <- unique(adjacency, MARGIN = 1)
  #inverse_indices <- match(as.matrix(adjacency), as.matrix(adjacency_unique))
  
  #np.dot(adjacency.astype(int), adjacency.T.astype(int)) == adjacency.sum(axis=-1, keepdims=True).T
  d = tcrossprod(adjacency * 1L)
  precedence <-  d == matrix(rowSums(adjacency * 1L), ncol=nrow(adjacency), nrow=nrow(d), byrow = TRUE)
  
  masks <- list()
  for (i in seq_along(c(hidden_features, out_features))) {
    if (i > 1) {
      mask <- precedence[, indices, drop = FALSE]
    } else {
      mask <- adjacency
    }
    
    if (all(!mask)) {
      stop("The adjacency matrix leads to a null Jacobian.")
    }
    
    if (i <= length(hidden_features)) {
      reachable <- which(rowSums(mask) > 0)
      if (length(reachable) > 0) {
        indices <- reachable[(seq_len(hidden_features[i]) - 1) %% length(reachable) + 1]
      } else {
        indices <- integer(0)
      }
      mask <- mask[indices, , drop = FALSE]
    } 
    #else {
    #  mask <- mask[inverse_indices, , drop = FALSE]
    #}
    masks[[i]] <- mask
  }
  return(masks)
}

if (FALSE){
  adjacency <- matrix(c(0, 1, 1, 0, 0, 0, 0, 0, 0), nrow = 3, byrow = FALSE)
  masks = create_masks(adjacency = adjacency, hidden_features = c(4,4))
  layer_sizes <- c(4, 4)
  
  input_layer <- layer_input(shape = list(3))
  h1 = LinearMasked(units=layer_sizes[1], mask=t(masks[[1]]))(input_layer)
  h1 = layer_activation(activation='relu')(h1)
  h2 = LinearMasked(units=layer_sizes[2], mask=t(masks[[2]]))(h1)
  h2 = layer_activation(activation='relu')(h2)
  out = LinearMasked(units=3L, mask=t(masks[[3]]))(h2)
  
  model = keras_model(inputs = input_layer, outputs = out)
  
  print(model)
  x = tf$ones(c(1L,3L))
  model(x)
  
  # Calculate the Jacobian matrix using Python and TensorFlow
  with(tf$GradientTape(persistent = TRUE) %as% tape, {
    tape$watch(x)
    y <- model(x)
  })
  
  tape$jacobian(y, x)
  
  
  
  hidden_features_r = c(4L,2L)
  adjacency <- matrix(c(0, 1, 1, 0, 0, 0, 0, 0, 0), nrow = 3, byrow = FALSE)
  masks = create_masks(adjacency = adjacency,hidden_features=hidden_features_r)
  dag_maf_plot(masks, c(3,4,2,3))
  
  library(reticulate)
  source_python("R/tram_dag/dag_maf.py")
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






