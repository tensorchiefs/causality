# Load necessary libraries
# library(tensorflow)
# library(keras)
# library(tfprobabiliy)
# library(sessioninfo)

# # Load necessary libraries
# library(reticulate)
# # If using reticulate, print Python configuration
# py_config()

# # Print TensorFlow and Keras version using Python
# py_run_string("import tensorflow as tf; print('TensorFlow version:', tf.__version__)")
# py_run_string("import keras; print('Keras version:', keras.__version__)")

# py_run_string("import tensorflow as tf; print('TensorFlow version:', tf.__version__)")

# # Print R session information
# sessionInfo()
# # Print version information for R packages
# installed.packages()[, c("Package", "Version")]
# Load necessary libraries
######## Oliver Python Env ####
#library(reticulate)
#use_virtualenv("/Users/oli/r_python_2024/", required = TRUE)
#library(tensorflow)
#install_tensorflow()
#reticulate::py_config()
#library(reticulate)
#tfp <- reticulate::import("tensorflow_probability")
#tf = reticulate::import("tensorflow")
#library(keras)
#use_condaenv("/Users/oli/miniforge3/envs/r-tensorflow", required = TRUE)


# Load necessary libraries
library(reticulate)
use_virtualenv("/Users/oli/r_python_2024/", required = TRUE)


reticulate::py_config()
#install.packages("tfprobability")

library(tensorflow)
library(tfprobability)
#install_tfprobability()
library(keras)

# Initialize TensorFlow and TensorFlow Probability
tf <- import('tensorflow')
library(tensorflow)
tfp <- import("tensorflow_probability")

# Define a normal distribution using TensorFlow Probability
normal_dist <- tfd$Normal(loc = 0, scale = 1)

# Evaluate the PDF at a point, for example, x = 1
pdf_value <- normal_dist$pdf(1) %>% tf$Session()$run()

# Print the result
print(pdf_value)