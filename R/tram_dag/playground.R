library(keras)
library(tensorflow)

Linear(keras$layers$Layer) %py_class% {
  initialize <- function(units = 32) {
    super$initialize()
    self$units <- units
  }
  
  build <- function(input_shape) {
    self$w <- self$add_weight(
      shape = shape(tail(input_shape, 1), self$units),
      initializer = "random_normal",
      trainable = TRUE
    )
    self$b <- self$add_weight(
      shape = shape(self$units),
      initializer = "random_normal",
      trainable = TRUE
    )
  }
  
  call <- function(inputs) {
    tf$matmul(inputs, self$w) + self$b
  }
}

x = tf$ones(c(2L, 2L))
linear_layer = Linear(4)
y = linear_layer(x)
