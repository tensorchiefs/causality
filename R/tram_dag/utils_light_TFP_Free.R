## This is a light version of the utils file, it is used in case no TFP is available 

R_START = 1-0.0001 #1.0-1E-1
L_START = 0.0001

library(tensorflow)
library(keras)
library(tidyverse)

version_info = function(){
  print(reticulate::py_config())
  print('R version:')
  print('tensorflow version:')
  print(tf$version$VERSION)
  print('keras version:')
  print(reticulate::py_run_string("import keras; print(keras.__version__)"))
  print('tfprobability version:')
  print(tfprobability::tfp_version())
}

is_upper_triangular <- function(mat) {
  # Ensure it's a square matrix
  if (nrow(mat) != ncol(mat)) {
    return(FALSE)
  }
  
  # Check if elements below the diagonal are zero
  for (i in 1:nrow(mat)) {
    for (j in 1:ncol(mat)) {
      if (j < i && mat[i, j] != 0) {
        return(FALSE)
      }
      if (j == i && mat[i, j] != 0) {
        return(FALSE)
      }
    }
  }
  
  return(TRUE)
}


unscale = function(dat_train_orig, dat_scaled){
  # Get original min and max
  orig_min = tf$reduce_min(dat_train_orig, axis=0L)
  orig_max = tf$reduce_max(dat_train_orig, axis=0L)
  dat_scaledtf = tf$constant(as.matrix(dat_scaled), dtype = 'float32')
  # Reverse the scaling
  return(dat_scaledtf * (orig_max - orig_min) + orig_min)
}

scale_df = function(dat_tf){
  dat_min = tf$reduce_min(dat_tf, axis=0L)
  dat_max = tf$reduce_max(dat_tf, axis=0L)
  dat_scaled = (dat_tf - dat_min) / (dat_max - dat_min)
  return(dat_scaled)
}

scale_validation = function(dat_training, dat_val){
  dat_min = tf$reduce_min(dat_training, axis=0L)
  dat_max = tf$reduce_max(dat_training, axis=0L)
  dat_scaled = (dat_val - dat_min) / (dat_max - dat_min)
  return(dat_scaled)
}

scale_value = function(dat_train_orig, col, value){
  # Get original min and max
  orig_min = tf$reduce_min(dat_train_orig[,col], axis=0L)
  orig_max = tf$reduce_max(dat_train_orig[,col], axis=0L)
  # Reverse the scaling
  return((value - orig_min) / (orig_max - orig_min))
}


# Function to calculate the SHA256 checksum
library(digest)
calculate_checksum <- function(weights) {
  # Start an empty vector to hold all byte-converted weights
  weights_bytes <- c()
  
  for (i in 1:length(weights)) {
    # Transform the weight to bytes
    weight_bytes <- serialize(weights[[i]], NULL)
    weights_bytes <- c(weights_bytes, weight_bytes)
  }
  
  # Calculate the digest on the concatenated byte strings
  return(digest(weights_bytes, algo="sha256", serialize=FALSE))
}

plot_obs_fit = function(parents_l, target_l, thetaNN_l,name){
"' Please note that this samples are generated using the **given observed** parents
"
  for (i in 1:length(parents_l)){
    parents = parents_l[[i]]
    targets = target_l[[i]]
    thetaNN = thetaNN_l[[i]]
    hist(targets$numpy(), freq = FALSE,100, main=paste0(name, ' x_',i, ' green for model'))
    x_samples = sample_from_target(thetaNN_l[[i]], parents)
    lines(density(x_samples$numpy()), col='green')
    #hist(x_samples$numpy(),100,xlim=c(0,1))
  }
}



























