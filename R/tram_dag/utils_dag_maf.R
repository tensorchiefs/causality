###### Builds a model #####
create_theta_tilde_maf = function(adjacency, order){
  input_layer <- layer_input(shape = list(ncol(adjacency)))
  outs = list()
  for (r in 1:order){
    d = input_layer
    for (i in 2:(length(layer_sizes) - 1)) {
      d = LinearMasked(units=layer_sizes[i], mask=t(masks[[i-1]]))(d)
      d = layer_activation(activation='relu')(d)
    }
    out = LinearMasked(units=layer_sizes[length(layer_sizes)], mask=t(masks[[length(layer_sizes) - 1]]))(d)
    outs = append(outs,out)
  }
  outs_c = keras$layers$concatenate(outs)
  outs_c2 = tf$transpose(tf$reshape(outs_c, shape = c(order, ncol(adjacency))))
  model = keras_model(inputs = input_layer, outputs = outs_c2)
  return(model)
}


###### LinearMasked ####
LinearMasked(keras$layers$Layer) %py_class% {
  
  initialize <- function(units = 32, mask = NULL) {
    super$initialize()
    self$units <- units
    self$mask <- mask  # Add a mask parameter
  }
  
  build <- function(input_shape) {
    self$w <- self$add_weight(
      name = "w",
      shape = shape(input_shape[[2]], self$units),
      initializer = "random_normal",
      trainable = TRUE
    )
    self$b <- self$add_weight(
      name = "b",
      shape = shape(self$units),
      initializer = "random_normal",
      trainable = TRUE
    )
    
    # Ensure the mask is the correct shape and convert it to a tensor
    if (!is.null(self$mask)) {
      if (!identical(dim(self$mask), dim(self$w))) {
        stop("Mask shape must match weights shape")
      }
      self$mask <- tf$convert_to_tensor(self$mask, dtype = self$w$dtype)
    }
  }
  
  call <- function(inputs) {
    if (!is.null(self$mask)) {
      # Apply the mask
      masked_w <- self$w * self$mask
    } else {
      masked_w <- self$w
    }
    tf$matmul(inputs, masked_w) + self$b
  }
}

###### Pure R Function ####
# Creates Autoregressive masks for a given adjency matrix and hidden features
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

# Load the required library
library(ggplot2)
library(grid)

# Function to draw the network
dag_maf_plot <- function(layer_masks, layer_sizes) {
  max_nodes <- max(layer_sizes)
  width <- max_nodes * 100
  min_x <- 0
  max_x <- width  # Adjust max_x to include input layer
  min_y <- Inf
  max_y <- -Inf
  
  # Create a data frame to store node coordinates
  nodes <- data.frame(x = numeric(0), y = numeric(0), label = character(0))
  
  # Draw the nodes for all layers
  for (i in 1:length(layer_sizes)) {
    size <- layer_sizes[i]
    layer_top <- max_nodes / 2 - size / 2
    
    for (j in 1:size) {
      x <- (i-1) * width
      y <- layer_top + j * 100
      label <- ifelse(i == 1, paste("x_", j, sep = ""), "")  # Add labels for the first column
      nodes <- rbind(nodes, data.frame(x = x, y = y, label=label))
      max_x <- max(max_x, x)
      min_y <- min(min_y, y)
      max_y <- max(max_y, y)
    }
  }
  
  # Create a data frame to store connection coordinates
  connections <- data.frame(x_start = numeric(0), y_start = numeric(0),
                            x_end = numeric(0), y_end = numeric(0))
  
  # Draw the connections
  for (i in 1:length(layer_masks)) {
    mask <- t(layer_masks[[i]])
    input_size <- nrow(mask)
    output_size <- ncol(mask)
    
    for (j in 1:input_size) {
      for (k in 1:output_size) {
        if (mask[j, k]) {
          start_x <- (i - 1) * width
          start_y <- max_nodes / 2 - input_size / 2 + j * 100
          end_x <- i * width
          end_y <- max_nodes / 2 - output_size / 2 + k * 100
          
          connections <- rbind(connections, data.frame(x_start = start_x, y_start = start_y,
                                                       x_end = end_x, y_end = end_y))
        }
      }
    }
  }
  
  
  # Create the ggplot object
  network_plot <- ggplot() +
    geom_segment(data = connections, aes(x = x_start, y = -y_start, xend = x_end, yend = -y_end),
                 color = 'black', size = 1,
                 arrow = arrow(type = "closed", length = unit(0.15, "inches"))) +
    geom_point(data = nodes, aes(x = x, y = -y), color = 'blue', size = 8,alpha = 0.5) +
    geom_text(data = nodes, aes(x = x, y = -y, label = label), vjust = 0, hjust = 0.5) +  # Add labels
    theme_void() 
  
  return(network_plot)
}




