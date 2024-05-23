###### Builds a model #####
create_theta_tilde_maf = function(adjacency, len_theta, layer_sizes){
  input_layer <- layer_input(shape = list(ncol(adjacency)))
  outs = list()
  for (r in 1:len_theta){
    d = input_layer
    for (i in 2:(length(layer_sizes) - 1)) {
      d = LinearMasked(units=layer_sizes[i], mask=t(masks[[i-1]]))(d)
      #d = layer_activation(activation='relu')(d)
      d = layer_activation(activation='sigmoid')(d)
    }
    out = LinearMasked(units=layer_sizes[length(layer_sizes)], mask=t(masks[[length(layer_sizes) - 1]]))(d)
    outs = append(outs,tf$expand_dims(out, axis=-1L)) #Expand last dim for concatenating
  }
  outs_c = keras$layers$concatenate(outs, axis=-1L)
  model = keras_model(inputs = input_layer, outputs = outs_c)
  return(model)
}

create_param_net <- function(len_param, input_layer, layer_sizes, masks, last_layer_bias=TRUE) {
  outs = list()
  for (r in 1:len_param){
    d = input_layer
    if (length(layer_sizes) > 2){ #Hidden Layers
      for (i in 2:(length(layer_sizes) - 1)) {
        d = LinearMasked(units=layer_sizes[i], mask=t(masks[[i-1]]))(d)
        #d = layer_activation(activation='relu')(d)
        d = layer_activation(activation='sigmoid')(d)
      }
    } #add output layers
    out = LinearMasked(units=layer_sizes[length(layer_sizes)], mask=t(masks[[length(layer_sizes) - 1]]),bias=last_layer_bias)(d)
    outs = append(outs,tf$expand_dims(out, axis=-1L)) #Expand last dim for concatenating
  }
  outs_c = keras$layers$concatenate(outs, axis=-1L)
}

# Creates a keras layer which takes as input (None, |x|) and returns (None, |x|, 1) which are all zero 
create_null_net <- function(input_layer) {
  output_layer <- layer_lambda(input_layer, function(x) {
    # Create a tensor of zeros with the same shape as x
    zeros_like_x <- k_zeros_like(x)
    # Add an extra dimension to match the desired output shape (None, |x|, 1)
    expanded_zeros_like_x <- k_expand_dims(zeros_like_x, -1)
    return(expanded_zeros_like_x)
  })
  return(output_layer)
}

# Creates a keras layer which takes as input (None, |x|) and returns (None, |x|, len) which are all constant variables


create_param_model = function(MA, hidden_features_I = c(2,2), len_theta=30, hidden_features_CS = c(2,2)){
  input_layer <- layer_input(shape = list(ncol(MA)))
 
  ##### Creating the Intercept Model
  if ('ci' %in% MA == TRUE) { # At least one 'ci' in model
    layer_sizes_I <- c(ncol(MA), hidden_features_I, nrow(MA))
    masks_I = create_masks(adjacency =  t(MA == 'ci'), hidden_features_I)
    h_I = create_param_net(len_param = len_theta, input_layer=input_layer, layer_sizes = layer_sizes_I, masks_I, last_layer_bias=TRUE)
    #dag_maf_plot(masks_I, layer_sizes_I)
    #model_ci = keras_model(inputs = input_layer, h_I)
  } else { # Adding simple interceps
    layer_sizes_I = c(ncol(MA), nrow(MA))
    masks_I = list(matrix(FALSE, nrow=nrow(MA), ncol=ncol(MA)))
    h_I = create_param_net(len_param = len_theta, input_layer=input_layer, layer_sizes = layer_sizes_I, masks_I, last_layer_bias=TRUE)
  }
    
  ##### Creating the Complex Shift Model
  if ('cs' %in% MA == TRUE) { # At least one 'cs' in model
    layer_sizes_CS <- c(ncol(MA), hidden_features_CS, nrow(MA))
    masks_CS = create_masks(adjacency =  t(MA == 'cs'), hidden_features_CS)
    h_CS = create_param_net(len_param = 1, input_layer=input_layer, layer_sizes = layer_sizes_CS, masks_CS, last_layer_bias=FALSE)
    #dag_maf_plot(masks_CS, layer_sizes_CS)
    # model_cs = keras_model(inputs = input_layer, h_CS)
  } else { #No 'cs' term in model --> return zero
    h_CS = create_null_net(input_layer)
  }
  
  ##### Creating the Linear Shift Model
  if ('ls' %in% MA == TRUE) {
  #h_LS = keras::layer_dense(input_layer, use_bias = FALSE, units = 1L)
      layer_sizes_LS <- c(ncol(MA), nrow(MA))
      masks_LS = create_masks(adjacency =  t(MA == 'ls'), c())
      out = LinearMasked(units=layer_sizes_LS[2], mask=t(masks_LS[[1]]), bias=FALSE, name='beta')(input_layer) 
      h_LS = tf$expand_dims(out, axis=-1L)#keras$layers$concatenate(outs, axis=-1L)
      #dag_maf_plot(masks_LS, layer_sizes_LS)
      #model_ls = keras_model(inputs = input_layer, h_LS)
  } else {
      h_LS = create_null_net(input_layer)
  }
  #Keras does not work with lists (only in eager mode)
  #model = keras_model(inputs = input_layer, outputs = list(h_I, h_CS, h_LS))
  #Dimensions h_I (B,3,30) h_CS (B, 3, 1) h_LS(B, 3, 3)
  # Convention for stacking
  # 1       CS
  # 2->|X|+1 LS
  # |X|+2 --> Ende M 
  outputs_tensor = keras$layers$concatenate(list(h_CS, h_LS, h_I), axis=-1L)
  param_model = keras_model(inputs = input_layer, outputs = outputs_tensor)
  return(param_model)
}


###### to_theta3 ####
# See zuko but fixed for order 3
to_theta3 = function(theta_tilde){
  shift = tf$convert_to_tensor(log(2) * dim(theta_tilde)[[length(dim(theta_tilde))]] / 2)
  order = tf$shape(theta_tilde)[3]
  widths = tf$math$softplus(theta_tilde[,, 2L:order, drop=FALSE])
  widths = tf$concat(list(theta_tilde[,, 1L, drop=FALSE], widths), axis = -1L)
  return(tf$cumsum(widths, axis = -1L) - shift)
}

### Bernstein Basis Polynoms of order M (i.e. M+1 coefficients)
# return (B,Nodes,M+1)
bernstein_basis <- function(tensor, M) {
  # Ensure tensor is a TensorFlow tensor
  tensor <- tf$convert_to_tensor(tensor)
  dtype <- tensor$dtype
  M = tf$cast(M, dtype)
  # Expand dimensions to allow broadcasting
  tensor_expanded <- tf$expand_dims(tensor, -1L)
  # Ensuring tensor_expanded is within the range (0,1) 
  tensor_expanded = tf$clip_by_value(tensor_expanded, tf$keras$backend$epsilon(), 1 - tf$keras$backend$epsilon())
  k_values <- tf$range(M + 1L) #from 0 to M
  
  # Calculate the Bernstein basis polynomials
  log_binomial_coeff <- tf$math$lgamma(M + 1.) - 
    tf$math$lgamma(k_values + 1.) - 
    tf$math$lgamma(M - k_values + 1.)
  log_powers <- k_values * tf$math$log(tensor_expanded) + 
    (M - k_values) * tf$math$log(1 - tensor_expanded)
  log_bernstein <- log_binomial_coeff + log_powers
  
  return(tf$exp(log_bernstein))
}


###### LinearMasked ####
LinearMasked(keras$layers$Layer) %py_class% {
  
  initialize <- function(units = 32, mask = NULL, bias=TRUE, name = NULL, trainable = NULL, dtype = NULL) {
    super$initialize(name = name)
    self$units <- units
    self$mask <- mask  # Add a mask parameter
    self$bias = bias
    # The additional arguments (name, trainable, dtype) are not used but are accepted to prevent errors during deserialization
  }
  
  build <- function(input_shape) {
    self$w <- self$add_weight(
      name = "w",
      shape = shape(input_shape[[2]], self$units),
      initializer = "random_normal",
      trainable = TRUE
    )
    if (self$bias) {
      self$b <- self$add_weight(
        name = "b",
        shape = shape(self$units),
        initializer = "random_normal",
        trainable = TRUE
      )
    } else{
      self$b <- NULL
    }
    
    # Handle the mask conversion if it's a dictionary (when loaded from a saved model)
    if (!is.null(self$mask)) {
      np <- import('numpy')
      if (is.list(self$mask) || "AutoTrackable" %in% class(self$mask)) {
        # Extract the mask value and dtype from the dictionary
        mask_value <- self$mask$config$value
        mask_dtype <- self$mask$config$dtype
        print("Hallo Gallo")
        mask_dtype = 'float32'
        print(mask_dtype)
        # Convert the mask value back to a numpy array
        mask_np <- np$array(mask_value, dtype = mask_dtype)
        # Convert the numpy array to a TensorFlow tensor
        self$mask <- tf$convert_to_tensor(mask_np, dtype = mask_dtype)
      } else {
        # Ensure the mask is the correct shape and convert it to a tensor
        if (!identical(dim(self$mask), dim(self$w))) {
          stop("Mask shape must match weights shape")
        }
        self$mask <- tf$convert_to_tensor(self$mask, dtype = self$w$dtype)
      }
    }
  }
  
  call <- function(inputs) {
    if (!is.null(self$mask)) {
      # Apply the mask
      masked_w <- self$w * self$mask
    } else {
      masked_w <- self$w
    }
    if(!is.null(self$b)){
      tf$matmul(inputs, masked_w) + self$b
    } else{
      tf$matmul(inputs, masked_w)
    }
  }
  
  get_config <- function() {
    config <- super$get_config()
    config$units <- self$units
    config$mask <- if (!is.null(self$mask)) tf$make_ndarray(tf$make_tensor_proto(self$mask)) else NULL
    config
  }
}


###### Pure R Function ####
# Creates Autoregressive masks for a given adjency matrix and hidden features
create_masks <- function(adjacency, hidden_features=c(64, 64)) {
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

########## Transformations ############

sample_standard_logistic <- function(shape, epsilon=1e-7) {
  uniform_samples <- tf$random$uniform(shape, minval=0, maxval=1)
  clipped_uniform_samples <- tf$clip_by_value(uniform_samples, epsilon, 1 - epsilon)
  logistic_samples <- tf$math$log(clipped_uniform_samples / (1 - clipped_uniform_samples))
  return(logistic_samples)
}


### h_dag
h_dag = function(t_i, theta){
  len_theta = tf$shape(theta)[3L] #TODO tied to 3er Tensors
  Be = bernstein_basis(t_i, len_theta-1L) 
  return (tf$reduce_mean(theta * Be, -1L))
}

### h_dag_dash
h_dag_dash = function(t_i, theta){
  len_theta = tf$shape(theta)[3L] #TODO tied to 3er Tensors
  Bed = bernstein_basis(t_i, len_theta-2L) 
  dtheta = (theta[,,2:len_theta,drop=FALSE]-theta[,,1:(len_theta-1L), drop=FALSE])
  return (tf$reduce_sum(dtheta * Bed, -1L))
}


h_dag_extra = function(t_i, theta){
  DEBUG = FALSE
  t_i3 = tf$expand_dims(t_i, axis=-1L)
  # for t_i < 0 extrapolate with tangent at h(0)
  b0 <- tf$expand_dims(h_dag(L_START, theta),axis=-1L)
  slope0 <- tf$expand_dims(h_dag_dash(L_START, theta), axis=-1L) 
  # If t_i < 0, use a linear extrapolation
  mask0 <- tf$math$less(t_i3, L_START)
  h <- tf$where(mask0, slope0 * (t_i3 - L_START) + b0, t_i3)
  #if (DEBUG) printf('~~~ eval_h_extra  Fraction of extrapolated samples < 0 : %f \n', tf$reduce_mean(tf$cast(mask0, tf$float32)))
  
  #(for t_i > 1)
  b1 <- tf$expand_dims(h_dag(R_START, theta),axis=-1L)
  slope1 <-  tf$expand_dims(h_dag_dash(R_START, theta), axis=-1L)
  # If t_i > 1, use a linear extrapolation
  mask1 <- tf$math$greater(t_i3, R_START)
  h <- tf$where(mask1, slope1 * (t_i3 - R_START) + b1, h)
  if (DEBUG) printf('~~~ eval_h_extra  Fraction of extrapolated samples > 1 : %f \n', tf$reduce_mean(tf$cast(mask1, tf$float32)))
  
  # For values in between, use the original function
  mask <- tf$math$logical_and(tf$math$greater_equal(t_i3, L_START), tf$math$less_equal(t_i3, R_START))
  h <- tf$where(mask, tf$expand_dims(h_dag(t_i, theta), axis=-1L), h)
  # Return the mean value
  return(tf$squeeze(h))
}

h_dag_extra_struc = function(t_i, theta, shift){
  DEBUG = FALSE
  if (length(t_i$shape) == 2) {
    t_i3 = tf$expand_dims(t_i, axis=-1L)
  } 
  # for t_i < 0 extrapolate with tangent at h(0)
  b0 <- tf$expand_dims(h_dag(L_START, theta) + shift,axis=-1L)
  slope0 <- tf$expand_dims(h_dag_dash(L_START, theta), axis=-1L) 
  # If t_i < 0, use a linear extrapolation
  mask0 <- tf$math$less(t_i3, L_START)
  h <- tf$where(mask0, slope0 * (t_i3 - L_START) + b0, t_i3)
  #if (DEBUG) printf('~~~ eval_h_extra  Fraction of extrapolated samples < 0 : %f \n', tf$reduce_mean(tf$cast(mask0, tf$float32)))
  
  #(for t_i > 1)
  b1 <- tf$expand_dims(h_dag(R_START, theta) + shift,axis=-1L)
  slope1 <-  tf$expand_dims(h_dag_dash(R_START, theta), axis=-1L)
  # If t_i > 1, use a linear extrapolation
  mask1 <- tf$math$greater(t_i3, R_START)
  h <- tf$where(mask1, slope1 * (t_i3 - R_START) + b1, h)
  if (DEBUG) printf('~~~ eval_h_extra  Fraction of extrapolated samples > 1 : %f \n', tf$reduce_mean(tf$cast(mask1, tf$float32)))
  
  # For values in between, use the original function
  mask <- tf$math$logical_and(tf$math$greater_equal(t_i3, L_START), tf$math$less_equal(t_i3, R_START))
  h <- tf$where(mask, tf$expand_dims(h_dag(t_i, theta) + shift, axis=-1L), h)
  # Return the mean value
  return(tf$squeeze(h))
}

h_dag_dash_extra = function(t_i, theta){
  t_i3 = tf$expand_dims(t_i, axis=-1L)
  slope0 <- tf$expand_dims(h_dag_dash(L_START, theta), axis=-1L) 
  mask0 <- tf$math$less(t_i3, L_START)
  h_dash <- tf$where(mask0, slope0, t_i3)
  
  slope1 <-  tf$expand_dims(h_dag_dash(R_START, theta), axis=-1L)
  mask1 <- tf$math$greater(t_i3, R_START)
  h_dash <- tf$where(mask1, slope1, h_dash)
  
  mask <- tf$math$logical_and(tf$math$greater_equal(t_i3, L_START), tf$math$less_equal(t_i3, R_START))
  h_dash <- tf$where(mask, tf$expand_dims(h_dag_dash(t_i,theta),axis=-1L), h_dash)
  return (tf$squeeze(h_dash))
}

dag_loss = function (t_i, theta_tilde){
  theta = to_theta3(theta_tilde)
  h_ti = h_dag_extra(t_i, theta)
  # The log of the logistic density at h is log(f(h))=−h−2log(1+e −h)
  # log_density2 = -h_ti - 2 * tf$math$log(1 + tf$math$exp(-h_ti))
  # Softpuls is nuerically more stable (according to ChatGPT) compared to log_density2
  log_latent_density = -h_ti - 2 * tf$math$softplus(-h_ti)
  h_dag_dashd = h_dag_dash_extra(t_i, theta)
  log_lik = log_latent_density + tf$math$log(tf$math$abs(h_dag_dashd))
  return (-tf$reduce_mean(log_lik))#(-tf$reduce_mean(log_lik, axis=-1L))
}



struct_dag_loss = function (t_i, h_params){
  # from the last dimension of h_params the first entriy is h_cs1
  # the second to |X|+1 are the LS
  # the 2+|X|+1 to the end is H_I
  h_cs <- h_params[,,1, drop = FALSE]
  h_ls <- h_params[,,2, drop = FALSE]
  theta_tilde <- h_params[,,3:dim(h_params)[3], drop = FALSE]
  #CI 
  theta = to_theta3(theta_tilde)
  h_I = h_dag_extra(t_i, theta)
  #LS
  h_LS = tf$squeeze(h_ls, axis=-1L)#tf$einsum('bx,bxx->bx', t_i, beta)
  #CS
  h_CS = tf$squeeze(h_cs, axis=-1L)
  
  h = h_I + h_LS + h_CS
  
  #Compute terms for change of variable formula
  log_latent_density = -h - 2 * tf$math$softplus(-h) #log of logistic density at h
  ## h' dh/dtarget is 0 for all shift terms
  log_hdash = tf$math$log(tf$math$abs(h_dag_dash_extra(t_i, theta)))
  
  log_lik = log_latent_density + log_hdash
  ### DEBUG 
  #if (sum(is.infinite(log_lik$numpy())) > 0){
  #  print("Hall")
  #}
  return (-tf$reduce_mean(log_lik))
}


dag_loss_dumm = function (t_i, theta_tilde){
  theta = to_theta3(theta_tilde)
  h_ti = h_dag_extra(t_i, theta)
  return (-tf$reduce_mean(h_ti, axis=-1L)) 
}

sample_logistics_within_bounds <- function(h_0, h_1) {
  samples <- map2_dbl(h_0, h_1, function(lower_bound, upper_bound) {
    while(TRUE) {
      sample <- as.numeric(tf$squeeze(sample_standard_logistic(c(1L,1L))))
      if (lower_bound < sample && sample < upper_bound) {
        return(sample)
      }
    }
  })
  return(samples)
}

##### Sampling from target ########
#' Draws 1-D samples from the defined target
#'
#' @param node the index of the target
#' @param parents (B,X) the tensor going into the param_model, note that due to the MAF=structure of 
#'                      the network only the parents have an effect  
#' @returns samples form the traget index
sample_from_target_MAF = function(param_model, node, parents){
  DEBUG_NO_EXTRA = FALSE
  theta_tilde = param_model(parents)
  #theta_tilde = tf$cast(theta_tilde, dtype=tf$float32)
  theta = to_theta3(theta_tilde)
  
  #h_0 =  tf$expand_dims(h_dag(L_START, theta), axis=-1L)
  #h_1 = tf$expand_dims(h_dag(R_START, theta), axis=-1L)
  h_0 =  h_dag(L_START, theta)
  h_1 = h_dag(R_START, theta)
  if (DEBUG_NO_EXTRA){
    s = sample_logistics_within_bounds(h_0$numpy(), h_1$numpy())
    latent_sample = tf$constant(s)
    stop("Not IMplemented") #latent_sample = latent_dist$sample(theta_tilde$shape[1])
  } else { #The normal case allowing extrapolations
    latent_sample = sample_standard_logistic(parents$shape)
  }
  object_fkt = function(t_i){
    return(h_dag_extra(t_i, theta) - latent_sample)
  }
  #object_fkt(t_i)
  #shape = tf$shape(parents)[1]
  #target_sample = tfp$math$find_root_chandrupatla(object_fkt, low = -1E5*tf$ones(c(shape,1L)), high = 1E5*tf$ones(c(shape,1L)))$estimated_root
  target_sample = tfp$math$find_root_chandrupatla(object_fkt, low = h_0, high = h_1)$estimated_root
  
  # Manuly calculating the inverse for the extrapolated samples
  ## smaller than h_0
  l = latent_sample#tf$expand_dims(latent_sample, -1L)
  mask <- tf$math$less_equal(l, h_0)
  #cat(paste0('~~~ sample_from_target  Fraction of extrapolated samples < 0 : %f \n', tf$reduce_mean(tf$cast(mask, tf$float32))))
  #tf$where(mask, beta_dist_h$prob(y_i)* theta_im, h)
  slope0 <- h_dag_dash(L_START, theta)#tf$expand_dims(h_dag_dash(L_START, theta), axis=-1L)
  target_sample = tf$where(mask, (l-h_0)/slope0, target_sample)
  
  ## larger than h_1
  mask <- tf$math$greater_equal(l, h_1)
  #tf$where(mask, beta_dist_h$prob(y_i)* theta_im, h)
  slope1<- h_dag_dash(R_START, theta)
  target_sample = tf$where(mask, (l-h_1)/slope1 + 1.0, target_sample)
  cat(paste0('sample_from_target Fraction of extrapolated samples > 1 : %f \n', tf$reduce_mean(tf$cast(mask, tf$float32))))
  return(target_sample[,node, drop=FALSE])
}

sample_from_target_MAF_struct = function(param_model, node, parents){
  DEBUG_NO_EXTRA = FALSE #Takes for ages not really working
  h_params = param_model(parents)
  
  h_cs <- h_params[,,1, drop = FALSE]
  h_ls <- h_params[,,2, drop = FALSE]
  theta_tilde <- h_params[,,3:dim(h_params)[3], drop = FALSE]
  theta = to_theta3(theta_tilde)
  h_LS = tf$squeeze(h_ls, axis=-1L)
  h_CS = tf$squeeze(h_cs, axis=-1L)
  
  #h_0_old =  tf$expand_dims(h_dag(L_START, theta), axis=-1L)
  #h_1 = tf$expand_dims(h_dag(R_START, theta), axis=-1L)
  h_0 =  h_LS + h_CS + h_dag(L_START, theta) #tf$expand_dims(h_LS + h_CS + h_dag(L_START, theta), axis=-1L)
  h_1 =  h_LS + h_CS + h_dag(R_START, theta) #tf$expand_dims(h_LS + h_CS + h_dag(R_START, theta), axis=-1L)
  if (DEBUG_NO_EXTRA){
    s = sample_logistics_within_bounds(h_0$numpy(), h_1$numpy())
    latent_sample = tf$constant(s)
    stop("Not IMplemented") #latent_sample = latent_dist$sample(theta_tilde$shape[1])
  } else { #The normal case allowing extrapolations
    latent_sample = sample_standard_logistic(parents$shape)
  }
  
  #t_i = tf$ones_like(h_LS) *0.5
  #h_dag_extra_struc(t_i, theta, shift = h_LS + h_CS)
  #h_dag_extra(t_i, theta)
  
  object_fkt = function(t_i){
    return(h_dag_extra_struc(t_i, theta, shift = h_LS + h_CS) - latent_sample)
  }
  #object_fkt(t_i)
  #shape = tf$shape(parents)[1]
  #target_sample = tfp$math$find_root_chandrupatla(object_fkt, low = -1E5*tf$ones(c(shape,1L)), high = 1E5*tf$ones(c(shape,1L)))$estimated_root
  target_sample = tfp$math$find_root_chandrupatla(object_fkt, low = h_0, high = h_1)$estimated_root
  
  # Manuly calculating the inverse for the extrapolated samples
  ## smaller than h_0
  l = latent_sample#tf$expand_dims(latent_sample, -1L)
  mask <- tf$math$less_equal(l, h_0)
  #cat(paste0('~~~ sample_from_target  Fraction of extrapolated samples < 0 : %f \n', tf$reduce_mean(tf$cast(mask, tf$float32))))
  #tf$where(mask, beta_dist_h$prob(y_i)* theta_im, h)
  slope0 <- h_dag_dash(L_START, theta)#tf$expand_dims(h_dag_dash(L_START, theta), axis=-1L)
  target_sample = tf$where(mask, (l-h_0)/slope0, target_sample)
  
  ## larger than h_1
  mask <- tf$math$greater_equal(l, h_1)
  #tf$where(mask, beta_dist_h$prob(y_i)* theta_im, h)
  slope1<- h_dag_dash(R_START, theta)
  target_sample = tf$where(mask, (l-h_1)/slope1 + 1.0, target_sample)
  cat(paste0('sample_from_target Fraction of extrapolated samples > 1 : %f \n', tf$reduce_mean(tf$cast(mask, tf$float32))))
  return(target_sample[,node, drop=FALSE])
}




do_dag = function(param_model, A, doX = c(0.5, NA, NA, NA), num_samples=1042){
  num_samples = as.integer(num_samples)
  N = length(doX) #NUmber of nodes
  
  #### Checking the input #####
  stopifnot(is_upper_triangular(A)) #A needs to be upper triangular
  stopifnot(param_model$input$shape[2L] == N) #Same number of variables
  stopifnot(nrow(A) == N)           #Same number of variables
  stopifnot(sum(is.na(doX)) >= N-1) #Currently only one Variable with do(might also work with more but not tested)
  
  # Looping over the variables assuming causal ordering
  #Sampling (or replacing with do) of the current variable x
  xl = list() 
  s = tf$ones(c(num_samples, N))
  for (i in 1:N){
    ts = NA
    parents = which(A[,i] == 1)
    if (length(parents) == 0) { #Root node?
      ones = tf$ones(shape=c(num_samples,1L),dtype=tf$float32)
      if(is.na(doX[i])){ #No do ==> replace with samples (conditioned on 1)
        ts = sample_from_target_MAF(param_model, i, s)
      } else{
        ts = doX[i] * ones #replace with do
      }
    } else { #No root node ==> the parents are present 
      if(is.na(doX[i])){ #No do ==> replace with samples (conditioned on 1)
        ts = sample_from_target_MAF(param_model, i, s)
      } else{ #Replace with do
        ones = tf$ones(shape=c(num_samples,1L),dtype=tf$float32) 
        ts = doX[i] * ones #replace with do
      }
    }
    #s[,i,drop=FALSE] = ts 
    mask <- tf$one_hot(indices = as.integer(i - 1L), depth = tf$shape(s)[2], on_value = 1.0, off_value = 0.0, dtype = tf$float32)
    # Adjust 'ts' to have the same second dimension as 's'
    ts_expanded <- tf$broadcast_to(ts, tf$shape(s))
    # Subtract the i-th column from 's' and add the new values
    s <- s - mask + ts_expanded * mask
  }
  return(s)
}

do_dag_struct = function(param_model, MA, doX = c(0.5, NA, NA, NA), num_samples=1042){
  num_samples = as.integer(num_samples)
  N = length(doX) #NUmber of nodes
  
  #### Checking the input #####
  stopifnot(is_upper_triangular(MA)) #MA needs to be upper triangular
  stopifnot(param_model$input$shape[2L] == N) #Same number of variables
  stopifnot(nrow(MA) == N)           #Same number of variables
  stopifnot(sum(is.na(doX)) >= N-1) #Currently only one Variable with do(might also work with more but not tested)
  
  # Looping over the variables assuming causal ordering
  #Sampling (or replacing with do) of the current variable x
  xl = list() 
  s = tf$ones(c(num_samples, N))
  for (i in 1:N){
    ts = NA
    parents = which(MA[,i] != "0")
    if (length(parents) == 0) { #Root node?
      ones = tf$ones(shape=c(num_samples,1L),dtype=tf$float32)
      if(is.na(doX[i])){ #No do ==> replace with samples (conditioned on 1)
        ts = sample_from_target_MAF_struct(param_model, i, s)
      } else{
        ts = doX[i] * ones #replace with do
      }
    } else { #No root node ==> the parents are present 
      if(is.na(doX[i])){ #No do ==> replace with samples (conditioned on 1)
        ts = sample_from_target_MAF_struct(param_model, i, s)
      } else{ #Replace with do
        ones = tf$ones(shape=c(num_samples,1L),dtype=tf$float32) 
        ts = doX[i] * ones #replace with do
      }
    }
    #We want to add the samples to the ith column i.e. s[,i,drop=FALSE] = ts 
    mask <- tf$one_hot(indices = as.integer(i - 1L), depth = tf$shape(s)[2], on_value = 1.0, off_value = 0.0, dtype = tf$float32)
    # Adjust 'ts' to have the same second dimension as 's'
    ts_expanded <- tf$broadcast_to(ts, tf$shape(s))
    # Subtract the i-th column from 's' and add the new values
    s <- s - mask + ts_expanded * mask
  }
  return(s)
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
                 arrow = arrow()) +
    geom_point(data = nodes, aes(x = x, y = -y), color = 'blue', size = 8,alpha = 0.5) +
    geom_text(data = nodes, aes(x = x, y = -y, label = label), vjust = 0, hjust = 0.5) +  # Add labels
    theme_void() 
  
  return(network_plot)
}




