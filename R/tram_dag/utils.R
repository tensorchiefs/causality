########### Functions ############

calc_NLL = function(nn_theta_tile, parents, target){
  theta_tilde = nn_theta_tile(parents)
  theta_tilde = tf$cast(theta_tilde, dtype=tf$float32)
  theta = to_theta(theta_tilde)
  latent = eval_h(theta, y_i = target, beta_dist_h = bp$beta_dist_h)
  h_dash = eval_h_dash(theta, target, beta_dist_h_dash = bp$beta_dist_h_dash)
  pz = tfd_logistic(loc=0, scale=1)
  return(
    -tf$math$reduce_mean(
      pz$log_prob(latent) + 
        tf$math$log(h_dash))
  )
}

predict_p_target = function(thetaNN, parents, target_grid){
  theta_tilde = thetaNN(parents)
  theta_tilde = tf$cast(theta_tilde, dtype=tf$float32)
  theta = to_theta(theta_tilde)
  latent = eval_h(theta, y_i = target_grid, beta_dist_h = bp$beta_dist_h)
  h_dash = eval_h_dash(theta, target_grid, beta_dist_h_dash = bp$beta_dist_h_dash)
  pz = tfd_logistic(loc=0, scale=1)
  p_target = pz$prob(latent) * h_dash
  return(p_target)
}

predict_h = function(thetaNN, parents, target_grid){
  theta_tilde = thetaNN(parents)
  theta_tilde = tf$cast(theta_tilde, dtype=tf$float32)
  theta = to_theta(theta_tilde)
  latent = eval_h(theta, y_i = target_grid, beta_dist_h = bp$beta_dist_h)
  return(latent)
}


sample_from_target = function(thetaNN, parents){
  theta_tilde = thetaNN(parents)
  theta_tilde = tf$cast(theta_tilde, dtype=tf$float32)
  theta = to_theta(theta_tilde)
  
  latent_sample = tfp$distributions$Logistic(loc=0, scale=1)$sample(theta_tilde$shape[1])
  #DELETE
  object_fkt = function(t_i){
    return(tf$reshape((eval_h(theta, y_i = t_i, beta_dist_h = bp$beta_dist_h) - latent_sample), c(2,1L)))
  }
  object_fkt = function(t_i){
    return(tf$reshape((eval_h(theta, y_i = t_i, beta_dist_h = bp$beta_dist_h) - latent_sample), c(theta_tilde$shape[1],1L)))
  }
  target_sample = tfp$math$find_root_chandrupatla(object_fkt, low = 0, high = 1)$estimated_root
  return(target_sample)
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

train_step = function(thetaNN, parents, target){
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