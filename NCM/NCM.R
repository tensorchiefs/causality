library(tensorflow)
library(keras)

# Define the MLP class
MLP <- R6::R6Class(
  "MLP",
  inherit = KerasModel,
  
  public = list(
    initialize = function(input_size, output_size, hidden_sizes) {
      model <- keras_model_sequential()
      
      hs <- c(input_size, hidden_sizes, output_size)
      for (i in 1:(length(hs) - 1)) {
        h0 <- hs[i]
        h1 <- hs[i + 1]
        model %>%
          layer_dense(units = h1, input_shape = c(h0)) %>%
          layer_activation("relu")
      }
      
      # Remove the last ReLU activation
      model$layers <- model$layers[-length(model$layers)]
      
      super$initialize(model)
    },
    
    forward = function(input) {
      self$model(input)
    }
  )
)

# Define the NCM class
NCM <- R6::R6Class(
  "NCM",
  inherit = ParameterizedSCM, # Assuming ParameterizedSCM is defined in R as well
  
  public = list(
    initialize = function(adj, scale = FALSE) {
      super$initialize(adj)
      
      # Define MLP for each variable in the graph
      for (V in self$graph) {
        V_name <- self$i2n(V)
        pa_V <- self$graph[[V]]
        hs <- c(10, 10, 10) # Assuming a fixed hidden layer size
        self$S[[V_name]] <- MLP$new(length(pa_V) + 1, 1, hs)
      }
    },
    
    forward = function(v, doX, Xi, samples, debug = FALSE) {
      Consistency <- tf$ones(c(samples, 1))
      Consistency <- tf$where(doX >= 0, tf$where(tf$tile(v[ , 1, drop = FALSE], c(samples, 1)) == doX, 1, 0), Consistency)
      
      pVs <- list()
      for (V in self$topology) {
        pa_V <- self$graph[[V]]
        
        V_arg <- do.call(tensorflow::tf$concat, list(c(lapply(pa_V, function(pa) tf$tile(v[, pa, drop = FALSE], c(samples, 1))), self$U[[self$i2n(V)]](samples)), axis = 1))
        
        pV <- tf$nn$sigmoid(self$S[[self$i2n(V)]]$forward(V_arg))
        pV <- pV * v[, V, drop = FALSE] + (1 - v[, V, drop = FALSE]) * (1 - pV)
        
        pV <- tf$where(Xi == V, tf$where(doX >= 0, tf$ones(c(samples, 1)), pV), pV)
        
        pVs <- append(pVs, list(pV))
      }
      
      pV <- do.call(tensorflow::tf$concat, c(list(Consistency), pVs, list(axis = 1)))
      
      agg <- function(t) {
        tf$reduce_mean(tf$reduce_prod(t, axis = 1))
      }
      
      ret <- agg(pV)
      
      if (debug) {
        browser()
      }
      
      ret
    }
  )
)
