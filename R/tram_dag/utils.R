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


#TODO change to eval_h extra (for evaluation of testset and maybe training)
calc_NLL = function(nn_theta_tile, parents, target){
  theta_tilde = nn_theta_tile(parents)
  theta_tilde = tf$cast(theta_tilde, dtype=tf$float32)
  theta = to_theta(theta_tilde)
  latent = eval_h(theta, y_i = target, beta_dist_h = bp$beta_dist_h)
  h_dash = eval_h_dash(theta, target, beta_dist_h_dash = bp$beta_dist_h_dash)
  pz = latent_dist
  return(
    -tf$math$reduce_mean(
      pz$log_prob(latent) + 
        tf$math$log(h_dash))
  )
}

#TODO change to eval_h extra
predict_p_target = function(thetaNN, parents, target_grid){
  theta_tilde = thetaNN(parents)
  theta_tilde = tf$cast(theta_tilde, dtype=tf$float32)
  theta = to_theta(theta_tilde)
  latent = eval_h(theta, y_i = target_grid, beta_dist_h = bp$beta_dist_h)
  h_dash = eval_h_dash(theta, target_grid, beta_dist_h_dash = bp$beta_dist_h_dash)
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


sample_from_target = function(thetaNN, parents){
  theta_tilde = thetaNN(parents)
  theta_tilde = tf$cast(theta_tilde, dtype=tf$float32)
  theta = to_theta(theta_tilde)
  h_0 =  tf$expand_dims(eval_h(theta, 0, beta_dist_h = bp$beta_dist_h), axis=1L)
  h_1 = tf$expand_dims(eval_h(theta, 1, beta_dist_h = bp$beta_dist_h), axis=1L)
  latent_sample = latent_dist$sample(theta_tilde$shape[1])
  
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
  l = tf$expand_dims(latent_sample, 1L)
  mask <- tf$math$less_equal(l, h_0)
  #tf$where(mask, beta_dist_h$prob(y_i)* theta_im, h)
  slope0 <- tf$expand_dims(eval_h_dash(theta, 0., bp$beta_dist_h_dash), axis=1L)
  target_sample = tf$where(mask, (l-h_0)/slope0, target_sample)
  
  mask <- tf$math$greater_equal(l, h_1)
  #tf$where(mask, beta_dist_h$prob(y_i)* theta_im, h)
  slope1<- tf$expand_dims(eval_h_dash(theta, 1., bp$beta_dist_h_dash), axis=1L)
  target_sample = tf$where(mask, (l-h_1)/slope1 + 1.0, target_sample)
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

train_step_old = function(thetaNN, parents, target){
  optimizer = tf$keras$optimizers$Adam(learning_rate=0.0001)
  with(tf$GradientTape() %as% tape, {
    NLL = calc_NLL(thetaNN, parents, target)
  })
  #Creating a list for all gradients
  n = 1
  tvars = list(thetaNN$trainable_variables) 
  #Calculation of the gradients
  grads = tape$gradient(NLL, tvars)
  for (i in 1:n){
    optimizer$apply_gradients(
      purrr::transpose(list(grads[[i]], tvars[[i]]))
    )  
  }
  return(NLL)
}