##################################
##### Utils for tram_dag #########
library(mlt)
library(tram)
library(MASS)
library(tensorflow)
library(keras)
library(tidyverse)
library(tfprobability)

source('tram_scm/bern_utils.R')
source('tram_scm/model_utils.R')

########### Functions ############
##### Creation of deep transformation models ######
# We need on for each variable in the SCM
split_data = function(A, dat_scaled.tf){
  parents_l = target_l = list()
  for (i in 1:ncol(A)){
    parents = which(A[,i] == 1)
    if (length(parents) == 0){ # No parents source node
      parents_tmp = 0*dat_scaled.tf[,i] + 1
      #thetaNN_tmp = make_model(len_theta, 1)
      target_tmp = dat_scaled.tf[,i, drop=FALSE]
    } else{ # Node has parents
      parents_tmp  = dat_scaled.tf[,parents,drop=FALSE]
      #thetaNN_tmp = make_model(len_theta, length(parents))
      target_tmp = dat_scaled.tf[,i, drop=FALSE]
    }
    parents_l = append(parents_l, parents_tmp)
    #thetaNN_l = append(thetaNN_l, thetaNN_tmp)
    target_l = append(target_l, target_tmp)
  }
  return(list(parents = parents_l, target = target_l))
}

make_thetaNN = function(A, parents_l){
  thetaNN_l = list()
  for (i in 1:length(parents_l)){
    parents = which(A[,i] == 1)
    if (length(parents) == 0){
      thetaNN_l[[i]] = make_model(len_theta, 1)
    } else{
      thetaNN_l[[i]] = make_model(len_theta, length(parents))
    }
  }
  return(thetaNN_l)
}


train_step = function(thetaNN_l, parents_l, target_l, optimizer){
  n = length(thetaNN_l)
  
  with(tf$GradientTape() %as% tape, {
    NLL = 0  # Initialize NLL
    for(i in 1:n) { # Assuming that thetaNN_l, parents_l and target_l have the same length
      NLL = NLL + calc_NLL(thetaNN_l[[i]], parents_l[[i]], target_l[[i]])
    }
  })
  
  tvars = list()
  for(i in 1:n) {
    tvars[[i]] = thetaNN_l[[i]]$trainable_variables
  }
  
  #Calculation of the gradients
  grads = tape$gradient(NLL, tvars)
  for (i in 1:n){
    optimizer$apply_gradients(
      purrr::transpose(list(grads[[i]], tvars[[i]]))
    )  
  }
  return(list(NLL=NLL))
}

create_directories_if_not_exist <- function(dir_path) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
    printf("Creating : %s \n", dir_path)
  }
}

calc_NLL = function(nn_theta_tile, parents, target){
  theta_tilde = nn_theta_tile(parents)
  theta_tilde = tf$cast(theta_tilde, dtype=tf$float32)
  theta = to_theta(theta_tilde)
  
  #latentold = eval_h(theta, y_i = target, beta_dist_h = bp$beta_dist_h)
  latent = eval_h_extra(theta, y_i = target, beta_dist_h = bp$beta_dist_h, beta_dist_h_dash = bp$beta_dist_h_dash)
  
  #h_dashOld = eval_h_dash(theta, target, beta_dist_h_dash = bp$beta_dist_h_dash)
  h_dash = eval_h_dash_extra(theta, target, beta_dist_h_dash = bp$beta_dist_h_dash)
  
  
  pz = latent_dist
  return(
    -tf$math$reduce_mean(
      pz$log_prob(latent) +
        tf$math$log(h_dash))
  )
  # return(
  #   -tfp$stats$percentile(
  #     pz$log_prob(latent) + 
  #       tf$math$log(h_dash), 50.)
  # )
}


predict_p_target = function(thetaNN, parents, target_grid){  theta_tilde = thetaNN(parents)
  theta_tilde = tf$cast(theta_tilde, dtype=tf$float32)
  theta = to_theta(theta_tilde)
  latent = eval_h_extra(theta, y_i = target_grid, beta_dist_h = bp$beta_dist_h, beta_dist_h_dash = bp$beta_dist_h_dash)
  h_dash = eval_h_dash_extra(theta, target_grid, beta_dist_h_dash = bp$beta_dist_h_dash)
  pz = latent_dist
  p_target = pz$prob(latent) * h_dash
  return(p_target)
}

predict_h = function(thetaNN, parents, target_grid){
  theta_tilde = thetaNN(parents)
  theta_tilde = tf$cast(theta_tilde, dtype=tf$float32)
  theta = to_theta(theta_tilde)
  latent = eval_h_extra(theta, y_i = target_grid, beta_dist_h = bp$beta_dist_h, beta_dist_h_dash= bp$beta_dist_h_dash)
  return(latent)
}

sample_within_bounds <- function(h_0, h_1) {
  samples <- map2_dbl(h_0, h_1, function(lower_bound, upper_bound) {
    while(TRUE) {
      sample <- as.numeric(latent_dist$sample())
      if (lower_bound < sample && sample < upper_bound) {
        return(sample)
      }
    }
  })
  return(samples)
}

sample_from_target = function(thetaNN, parents){
  theta_tilde = thetaNN(parents)
  theta_tilde = tf$cast(theta_tilde, dtype=tf$float32)
  theta = to_theta(theta_tilde)
  h_0 =  tf$expand_dims(eval_h(theta, L_START, beta_dist_h = bp$beta_dist_h), axis=1L)
  h_1 = tf$expand_dims(eval_h(theta, R_START, beta_dist_h = bp$beta_dist_h), axis=1L)
  if (DEBUG_NO_EXTRA){
    s = sample_within_bounds(h_0$numpy(), h_1$numpy())
    latent_sample = tf$constant(s)
    if(FALSE){
      h_0 =  tf$expand_dims(eval_h(theta, 0.01, beta_dist_h = bp$beta_dist_h), axis=1L)
      h_1 = tf$expand_dims(eval_h(theta, 0.99, beta_dist_h = bp$beta_dist_h), axis=1L)
      h_0_2 = tf$squeeze(tf$concat(c(h_0, h_0), axis=0L))
      h_1_2 = tf$squeeze(tf$concat(c(h_1, h_1), axis=0L))
      len = as.numeric(theta_tilde$shape[1])
      l = latent_dist$sample(h_0_2$shape[1])
      # Get the boolean mask where condition is true
      mask = tf$math$logical_and(l >= h_0_2, l <= h_1_2)
      
      
      
      # Use boolean mask to get the values
      latent_sample = tf$boolean_mask(l, mask)[1:len]
    }
    
    
  } else { #The normal case allowing extrapolations
    latent_sample = latent_dist$sample(theta_tilde$shape[1])
  }
  
  #  object_fkt = function(t_i){
  #     return(tf$reshape((eval_h(theta, y_i = t_i, beta_dist_h = bp$beta_dist_h) - latent_sample), c(theta_tilde$shape[1],1L)))
  # }
  # shape = tf$shape(parents)[1]
  # target_sample1 = tfp$math$find_root_chandrupatla(object_fkt, low = 0, high = 1)$estimated_root
  # target_sample1
  
  object_fkt = function(t_i){
    return(tf$reshape((eval_h_extra(theta, y_i = t_i, beta_dist_h = bp$beta_dist_h,beta_dist_h_dash = bp$beta_dist_h_dash) - latent_sample), c(theta_tilde$shape[1],1L)))
  }
  shape = tf$shape(parents)[1]
  #target_sample = tfp$math$find_root_chandrupatla(object_fkt, low = -1E5*tf$ones(c(shape,1L)), high = 1E5*tf$ones(c(shape,1L)))$estimated_root
  target_sample = tfp$math$find_root_chandrupatla(object_fkt, low = h_0, high = h_1)$estimated_root
  
  # Manuly calculating the inverse for the extrapolated samples
  ## smaller than h_0
  l = tf$expand_dims(latent_sample, 1L)
  mask <- tf$math$less_equal(l, h_0)
  printf('~~~ sample_from_target  Fraction of extrapolated samples < 0 : %f \n', tf$reduce_mean(tf$cast(mask, tf$float32)))
  #tf$where(mask, beta_dist_h$prob(y_i)* theta_im, h)
  slope0 <- tf$expand_dims(eval_h_dash(theta, 0., bp$beta_dist_h_dash), axis=1L)
  target_sample = tf$where(mask, (l-h_0)/slope0, target_sample)
  
  ## larger than h_1
  mask <- tf$math$greater_equal(l, h_1)
  #tf$where(mask, beta_dist_h$prob(y_i)* theta_im, h)
  slope1<- tf$expand_dims(eval_h_dash(theta, 1., bp$beta_dist_h_dash), axis=1L)
  target_sample = tf$where(mask, (l-h_1)/slope1 + 1.0, target_sample)
  printf('sample_from_target Fraction of extrapolated samples > 1 : %f \n', tf$reduce_mean(tf$cast(mask, tf$float32)))
  return(target_sample)
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


make_model = function(len_theta, parent_dim){ 
  model <- keras_model_sequential() 
  model %>% 
    layer_dense(units=(10), input_shape = c(parent_dim), activation = 'tanh') %>% 
    layer_dense(units=(100), activation = 'tanh') %>% 
    layer_dense(units=len_theta) %>% 
    layer_activation('linear') 
  return (model)
}

do_training = function(name, thetaNN_l, train_data, val_data, SUFFIX, epochs=200, optimizer= tf$keras$optimizers$Adam(learning_rate=0.001)){
  parents_l = train_data$parents
  target_l = train_data$target

  parents_l_val = val_data$parents
  target_l_val = val_data$target
  
  
  loss = loss_val = rep(NA, epochs)
  dirname = paste0(DROPBOX, "exp/", train$name, "/", SUFFIX, "/")
  create_directories_if_not_exist(dirname)
  loss_name = paste0(dirname, 'losses.rda')
  
  if(file.exists(loss_name) == FALSE){
    for (e in 1:epochs){
      l = train_step(thetaNN_l, parents_l, target_l, optimizer=optimizer)
      loss[e]  = l$NLL$numpy()
      if (e %% 5 == 0) {
        NLL_val = 0  
        for(i in 1:ncol(val$A)) { # Assuming that thetaNN_l, parents_l and target_l have the same length
          NLL_val = NLL_val + calc_NLL(thetaNN_l[[i]], parents_l_val[[i]], target_l_val[[i]])$numpy()
        }
        loss_val[e] = NLL_val
        printf("e:%f  Train: %f, Val: %f \n",e, l$NLL$numpy(), NLL_val)
        for (i in 1:ncol(val$A)){
          printf('Layer %d checksum: %s \n',i, calculate_checksum(thetaNN_l[[i]]$get_weights()))
          
          #There might be a problem with h5
          #fn = paste0(dirname,train$name, "_nn", i, "_e", e, "_weights.h5")
          #thetaNN_l[[i]]$save_weights(path.expand(fn))
          
          fn = paste0(dirname,train$name, "_nn", i, "_e", e, "_weights.rds")
          saveRDS(thetaNN_l[[i]]$get_weights(), path.expand(fn))
          
          #fn = paste0(dirname,train$name, "_nn", i, "_e", e, "_model.h5")
          #save_model_hdf5(thetaNN_l[[i]], fn)
          
          fn = paste0(dirname,train$name, "_nn", i, "_e", e, "_checksum.txt")
          write(calculate_checksum(thetaNN_l[[i]]$get_weights()), fn)
        }
      }
    }
    # Saving weights after training
    save(loss, loss_val, file=loss_name)
  }  else {
    print('Loss File Exists')
  }
  load(loss_name)
  return(list(loss, loss_val))
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


