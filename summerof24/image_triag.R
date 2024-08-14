# Execute oliver minimal

library(reticulate)
reticulate::py_config()
library(tensorflow)
library(keras)

set.seed(1)
fn = 'summerof24/image_keras_triag.h5' 


##################################
##### Utils for tram_dag #########
library(mlt)
library(tram)
library(MASS)
library(tensorflow)
library(keras)
library(tidyverse)
source('summerof24/utils_tf.R')

#### For TFP
library(tfprobability)
source('summerof24/utils_tfp.R')

############## Semi-structured DAG with image (Very Special to DAG in this file) ################
semi_struct_dag_loss = function (t_i, h_params){
  
  k_min <- k_constant(global_min)
  k_max <- k_constant(global_max)
  
  
  #### This function is used for the semi-structured model in which an image is included
  h_params_tabular = h_params[,,1:(dim(h_params)[3]-1)]
  h_params_image = h_params[,,dim(h_params)[3]]
  
  h_cs <- h_params_tabular[,,1, drop = FALSE]
  h_ls <- h_params_tabular[,,2, drop = FALSE]
  theta_tilde <- h_params_tabular[,,3:dim(h_params_tabular)[3], drop = FALSE]
  
  #Intercept 
  theta = to_theta3(theta_tilde)
  h_I = h_dag_extra(t_i, theta, k_min, k_max)  # batch, #Variables
  #LS
  h_LS = tf$squeeze(h_ls, axis=-1L)#tf$einsum('bx,bxx->bx', t_i, beta)
  #CS
  h_CS = tf$squeeze(h_cs, axis=-1L)  # batch, #Variables
  
  # CNN-eta(B)
  h_eta_B = h_params_image  # batch, #Variables
  
  h = h_I + h_LS + h_CS + h_eta_B
  
  #Compute terms for change of variable formula
  log_latent_density = -h - 2 * tf$math$softplus(-h) #log of logistic density at h
  ## h' dh/dtarget is 0 for all shift terms
  log_hdash = tf$math$log(tf$math$abs(h_dag_dash_extra(t_i, theta, k_min, k_max)))
  
  log_lik = log_latent_density + log_hdash
  ### DEBUG 
  #if (sum(is.infinite(log_lik$numpy())) > 0){
  #  print("Hall")
  #}
  return (-tf$reduce_mean(log_lik))
}

### Loading CIFAR10 data
#my_images <- dataset_cifar10()
my_images <- dataset_mnist()

## Getting Training data
n_obs = 60000
train_images <- my_images$train$x[1:n_obs,,] / 255.
train_labels <- my_images$train$y[1:n_obs]
dim(train_images)

# Getting the test data
test_images <- my_images$test$x / 255.
test_labels <- my_images$test$y
dim(test_images)

if (FALSE){
  ###### Hack Attack replace the images with the first image of the label
  idx = rep(NA, 10)
  for (i in 1:10){
    idx[i] = which(train_labels == i-1)[1]
  }
  train_labels[idx] #0 1 2 3 4 5 6 7 8 9
  # Get the index of the image in which the label the first time appreas
  for (i in 1:n_obs){
    #i = 1000
    id = idx[train_labels[i] + 1]
    train_images[i,,] = train_images[id,,] 
  }
}

plot_images <- function(img_array, main_label) {
  # Convert the image data to a format suitable for rasterImage (adjust dimensions)
  # Normalize the pixel values to [0, 1]
  img_array <- img_array 
  # Prepare plotting area and plot the image
  plot(0:1, 0:1, type = "n", xlab = "", ylab = "", main = main_label, axes = FALSE)
  rasterImage(img_array, 0, 0, 1, 1)
}

# Setting up the plot area
par(mfrow = c(2, 5), mar = c(1, 1, 2, 1))
# Loop through the first 10 images
for (i in c(1:120)) {
  #i=100
  print(i)
  # Reshape each image (keeping the color channel as the last dimension)
  img <- train_images[i,,] 
  # Determine label
  label <- train_labels[i] 
  # Plot the image
  plot_images(img, main_label = paste("Label:", label))
}
# Reset par to default
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)

##### TEMP
dgp <- function(n_obs, b_labels) {
    scale_effect_size = 10.
    
    # Create X1 depending on labels of images 
    # Images <=5 are "female" the other "male" with noise
    x1 = ifelse(b_labels + runif(n_obs,-2,2) < 5, 0, 1) 
    #x1 = sample(c(0,1),size=n_obs,replace = TRUE) 
    ####### x2 
    u2 = rlogis(n_obs, location =  0, scale = 1) 
    b2 = 0.5 * scale_effect_size
    #h_0(x2) = 0.42 * x2
    #u2 = h(x2 | x1) = h_0(x2) + b2 * x1 = 0.42 * x2 + 0.5 * x1
    x2 = (u2 - b2*x1)/0.42
    
    ####### x3
    u3 = rlogis(n_obs, location =  0, scale = 1) 
    a1 = 0.2  *  scale_effect_size 
    a1 = 0.9  *  scale_effect_size  #<--- New
    a2 = 0.03 *  scale_effect_size 
    #u3 = h(x3|x1,x2,B) = h_0(x3) + eta(B) + a1*x1 + a2*x2
    #h_0(x3) = 0.21 * x3 
    etaB = (b_labels - 4) #
    #etaB = 0 #No Effect 
    etaB = sqrt(b_labels) #Effect
    #Hack Attack
    #etaB = 0 --> Coefficients work
    #etaB = sqrt(b_labels) #Effect 
    x3 =  (u3 - etaB - a1*x1 - a2*x2)/0.21
    
    #Orginal Data
    dat =  data.frame(x1 = x1, x2 = x2, x3 = x3)
    dat.tf = tf$constant(as.matrix(dat), dtype = 'float32')
    
    # Minimal values
    q1 = quantile(dat[,1], probs = c(0.05, 0.95)) 
    q2 = quantile(dat[,2], probs = c(0.05, 0.95))
    q3 = quantile(dat[,3], probs = c(0.05, 0.95))
    
    A <- matrix(c(0, 1, 1, 0,0,1,0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
    return(list(
      df_orig=dat.tf,  
      min = tf$constant(c(q1[1], q2[1], q3[1]), dtype = 'float32'),
      max = tf$constant(c(q1[2], q2[2], q3[2]), dtype = 'float32'),
    A=A))
} 


# compare to Colr for tabular part
train = dgp(n_obs=n_obs, b_labels = train_labels[1:n_obs])
global_min = train$min
global_max = train$max
test = dgp(n_obs=10000, b_labels = test_labels)


####### COLR #######
if (FALSE){
  df = as.data.frame(train$df_orig$numpy())
  colnames(df) = c('X1', 'X2', 'X3')
  df$X4 = sqrt(train_labels[1:n_obs])
  lm(X4 ~ X1, df)
  confint(Colr(X3 ~ X1 + X2 + X4,df))
  confint(Colr(X2 ~ X1,df))
  
  # Fitting Tram
  df = data.frame(train$df_orig$numpy())
  pairs(df)
  
  
  fit.X2 = Colr(X2~X1,df)
  summary(fit.X2)  # sollte 0.5 sein, ist auch
  confint(fit.X2)
  
  df$label = train_labels[1:n_obs] 
  fit.X3 = Colr(X3 ~ X1 + X2 + label,df)
  summary(fit.X3) 
  
  fit.X3_X1 = Colr(X3 ~ X1 + X2 + label,df)
  summary(fit.X3) 
} # end Colr
#Bis jetzt alles CI
#MA =  matrix(c(0, 'ls', 'ci', 0,0,'cs',0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
#MA =  matrix(c(0, 'ls', 'ls', 0,0,'ls',0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
MA =  matrix(c(0, 'ls', 'ls', 0,0, 'ls',0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
MA
hidden_features_I = c(2,2)
hidden_features_CS = c(2,2)
len_theta = 6
tabular_model = create_param_model(MA, hidden_features_I = hidden_features_I, 
                                 len_theta = len_theta, 
                                 hidden_features_CS = hidden_features_CS)

#summary(tabular_model)
tabular_out = tabular_model$output
tabular_out$shape
# batch, no Variable, 1(CS) + 1(LS) + M
# params pro Variable
# first CS, second LS, third part BP-coeffs
tabular_model(train$df_orig)

###### create a cnn model ####
input_shape <- c(28, 28, 1)

# Define the CNN model using the functional API
cnn_input <- layer_input(shape = input_shape)

conv1 <- layer_conv_2d(filters = 8, kernel_size = c(3, 3), activation = 'relu')(cnn_input)
pool1 <- layer_max_pooling_2d(pool_size = c(2, 2))(conv1)
dropout1 <- layer_dropout(rate = 0.35)(pool1)

conv2 <- layer_conv_2d(filters = 16, kernel_size = c(3, 3), activation = 'relu')(dropout1)
pool2 <- layer_max_pooling_2d(pool_size = c(2, 2))(conv2)

dropout2 <- layer_dropout(rate = 0.35)(pool2)

flatten <- layer_flatten()(dropout2)

dense1 <- layer_dense(units = 30, activation = 'relu')(flatten)
dropout <- layer_dropout(rate = 0.35)(dense1)

cnn_out <- layer_dense(units = 1, activation = 'linear')(dropout)

# get shape batches, #Variables, 
# since image impacts only x3, the first 2 cols should be zero
repeated_tensor <- layer_repeat_vector(n = 3)(cnn_out)
tmp = k_reshape(k_constant(c(0,0,1)), shape=c(-1,3,1))
cnn_tensor <- layer_multiply(repeated_tensor, tmp)
# Reshape the tensor
cnn_out2 <- layer_reshape(target_shape = c(3, 1))(cnn_tensor)

merged_output <- layer_concatenate(list( tabular_out, cnn_out2))
merged_output$shape
# [batch, #variable, 1(CS) + 1(LS) + M(BP) + 1(CNN) ]

# Define final model
final_model <- keras_model(inputs = list(tabular_model$input, cnn_input), 
                           outputs = merged_output)

#summary(final_model)

h_params = final_model(list(train$df_orig, train_images ))
h_params$shape
# [batch, #variable, 1(CS) + 1(LS) + M(BP) + 1(CNN) ]

# check that only third variable has an eta_B Shift (last h_params-dim)
h_params[, ,9]


# loss before training
# since images are in a source node we do not need to provide them
semi_struct_dag_loss(train$df_orig, h_params)

##### Optimizer ####
if (FALSE){
  library(deepregression)
  print(final_model)
  
  get_optimizer = function(layer){
    if (layer$name == 'beta'){
      return(optimizer_adam(learning_rate=1e-2))
    } else {
      if (layer$trainable){
        return(optimizer_adam(learning_rate=1e-4))
      } else {
        return(NULL)
      }
    }
  }
  
  layers = final_model$layers
  ol = list()
  for (i in 1:length(layers)){
    #if (layers[[i]]$name == 'beta'){
      print(layers[[i]]$name)
      ol = append(ol, tuple(get_optimizer(layers[[i]]), layers[[i]]))
    #}
  }

if(FALSE){
  ol = list(
    tuple(optimizer_adam(learning_rate=1e-4), get_layer(final_model, 'dense_1'))
  )
}

optimizer = deepregression::multioptimizer(ol)
class(optimizer)
optimizer$minimize()
class(optimizer_adam(learning_rate=1e-4))
} else{
  optimizer = optimizer_adam()
}
optimizer
final_model$compile(optimizer, loss=semi_struct_dag_loss)
final_model$evaluate(x = list(train$df_orig, train_images), y=train$df_orig, batch_size = 7L)
final_model

##### Training with Hessian ####
if (FALSE){

if (FALSE){
  # Train the model using a custom training loop (just to check)
  # Custom training loop w/o second order 
  #4.17 --> 4.12
  for (epoch in 1:2) {
    with(tf$GradientTape(persistent = TRUE) %as% tape, {
      # Compute the model's prediction - forward pass
      h_params <- final_model(list(train$df_orig, train_images))
      loss <- semi_struct_dag_loss(train$df_orig, h_params)
    })
    # Compute gradients
    gradients <- tape$gradient(loss, final_model$trainable_variables)
    # Apply gradients to update the model parameters
    optimizer$apply_gradients(purrr::transpose(list(gradients, final_model$trainable_variables)))
    # Print the loss every epoch or more frequently if desired
    print(paste("Epoch", epoch, ", Loss:", loss$numpy()))
  }
}

beta_weights <- final_model$get_layer(name = "beta")$weights[[1]]

# Custom training loop
# Separate the weights of "beta" layers
# Retrieve the specific beta weights variable
## Training in which the beta layers are trained with a 2nd order update using a Hessian
train_step_with_hessian_beta_semi_struct <- tf_function(autograph = TRUE, 
                                            function(train_data, train_images, beta_weights, optimizer, lr_hessian = 0.1, beta_only=FALSE) {
#train_step <- function(train_data, beta_weights) {
    # train_data = train$df_orig
    #with(tf$GradientTape(persistent = TRUE) %as% tape2, { # Gradients for second-order derivatives
    #  with(tf$GradientTape(persistent = TRUE) %as% tape1, { # Gradients for first-order derivatives 
    # if (beta_only) {
    #   
    # } else {
    #   
    # }
    with(tf$GradientTape() %as% tape2, { # Gradients for second-order derivatives
      with(tf$GradientTape() %as% tape1, { # Gradients for first-order derivatives 
        h_params <- final_model(list(train_data, train_images))
        loss <- semi_struct_dag_loss(train_data, h_params)
        #hist = param_model$fit(x = train$df_orig, y=train$df_orig, epochs = 500L,verbose = TRUE)
      })
    
      # Compute first-order gradients
      all_weights <- final_model$trainable_weights
      all_grads <- tape1$gradient(loss, all_weights)
      #optimizer$apply_gradients(purrr::transpose(list(all_grads, all_weights))) #HACKATTACK
      other_gradients <- all_grads[!sapply(final_model$trainable_weights, function(weight) {
        identical(weight$name, beta_weights$name)
      })]
      beta_gradients <- all_grads[sapply(final_model$trainable_weights, function(weight) {
        identical(weight$name, beta_weights$name)
      })]
      other_weights <- final_model$trainable_weights[!sapply(final_model$trainable_weights, function(weight) {
        identical(weight$name, beta_weights$name)
      })]
      if (length(beta_gradients) != 1) {
        stop("Current implementation only supports **one** beta layer")
      }
      b = beta_gradients[[1]]
      bl_shape <- beta_weights$shape
      hessians <- tape2$jacobian(beta_gradients[[1]], beta_weights)  
   }) 
  optimizer$apply_gradients(purrr::transpose(list(other_gradients, other_weights))) 
  # Manipulate gradients and apply them 
    # Flatten the Hessian tensor to a matrix for inversion
    hessian_size <- bl_shape[[1]] * bl_shape[[2]]
    hessian_flat <- tf$reshape(hessians, shape = c(hessian_size, hessian_size))  # Adjust shape as needed
    # Add regularization to the Hessian
    hessian_flat <- hessian_flat + tf$eye(hessian_size) * 1e-8
    # Compute the inverse of the Hessian matrix
    hessian_inv <- tf$linalg$inv(hessian_flat)
    # DEBUG HACK ATTACK - replace with tf$linalg$inv when fixed <-------------TODO 
    #hessian_inv <- tf$eye(hessian_size)  # Identity matrix of appropriate size
    # Flatten the gradient for matrix multiplication
    grads_flat <- tf$reshape(beta_gradients[[1]], shape = c(hessian_size, 1L))
    # Compute the update using Hessian and gradient (this is the newton update rule)
    beta_update <- tf$matmul(hessian_inv, grads_flat)
    # Reshape the update back to the original shape
    beta_update_reshaped <- tf$reshape(beta_update, shape = bl_shape)  # Adjust shape as needed
    # Apply the update to the beta weights with the learning rate
    beta_weights$assign_sub(lr_hessian * beta_update_reshaped)
  loss
})
optimizer <- optimizer_adam()
  
optimizer <- tf$keras$optimizers$Adam()  # Adam optimizer for other layers)
# Wrap the custom training loop with tf_function
# Prepare the dataset with batching
batch_size <- 32
num_batches <- ceiling(nrow(train$df_orig) / batch_size)
indices <- sample(nrow(train$df_orig)) # Shuffle the indices

# Custom training loop with batches
epochs <- 50
loss_values <- numeric(epochs)  # Vector to store loss values for each epoch
loss_val <- numeric(epochs)

# Lists to store beta weights for each epoch
betas_21 <- numeric(epochs)
betas_31 <- numeric(epochs)
betas_32 <- numeric(epochs)

time.start = Sys.time()
time.last = Sys.time()
for (epoch in 1:epochs) {
  #epoch = 1
  batch_losses <- c()  # Vector to store loss values for the current epoch's batches
  
  for (batch_num in 1:num_batches) {
    #batch_num = 1
    start_index <- (batch_num - 1) * batch_size + 1
    end_index <- min(batch_num * batch_size, nrow(train$df_orig))
    if (start_index > end_index) {
      next
    }
    batch_indices <- indices[start_index:end_index]
    if (length(batch_indices) == 0) {
      next
    }
    batch_indices_tf <- tf$constant(batch_indices - 1L, dtype = tf$int32)
    batch_data <- tf$gather(train$df_orig, batch_indices_tf)
    # needs 
    loss <- train_step_with_hessian_beta_semi_struct(train_data = batch_data,
                                         train_images = train_images[batch_indices,,],
                                         beta_weights = beta_weights, 
                                         optimizer = optimizer,
                                         lr_hessian = 0.1)
    # Collect the batch loss
    batch_losses <- c(batch_losses, loss$numpy())
  }
  
  # Calculate the average loss for the current epoch
  avg_epoch_loss <- mean(batch_losses)
  
  # Store the average loss for the current epoch
  loss_values[epoch] <- avg_epoch_loss
  
  # Extract and store the beta weights for the current epoch
  betas_out <- final_model$get_layer(name = "beta")$get_weights() * final_model$get_layer(name = "beta")$mask
  betas_out <- betas_out[1,,]$numpy()
  betas_21[epoch] <- betas_out[1,2]
  betas_31[epoch] <- betas_out[1,3]
  betas_32[epoch] <- betas_out[2,3]
  
  validation_data = list(list(test$df_orig, test_images))
  val_loss = final_model$evaluate(x = list(test$df_orig, test_images), y=test$df_orig)
  loss_val[epoch] <- val_loss
  
  if (epoch %% 10 == 0 || epoch == 1) {
    #Print time needed for 10 epochs
    print(Sys.time())
    print(Sys.time() - time.last)
    time.last = Sys.time()
    cat(sprintf("Epoch %d, Average Loss: %f val_loss: %f)\n", epoch, avg_epoch_loss, val_loss))
    print(paste0('Betas: ', betas_out[1,2]))#, ' colr:', b21.colr))
    print(paste0('Betas: ', betas_out[1,3]))#, ' colr:', b31.colr))
    print(paste0('Betas: ', betas_out[2,3]))#, ' colr:', b32.colr))
  }
}

# After training, you can inspect the loss values and beta weights
print(loss_values)
print(betas_21)
print(betas_31)
print(betas_32)

# Plot the loss value
#loss_values_2 = loss_values
#loss_val_2 = loss_val
plot(loss_values, type = "l", xlab = "Epoch", ylab = "Loss", main = "Semi Hessian Optimization lr_b = 0.1 (two runs)", ylim=c(-0.5,-0.4))
lines(loss_val, col='green')
lines(loss_values_2, col='black')
lines(loss_val_2, col='green')

# Plot the beta weights together with the Colr estimates in a single plot
plot(betas_21, type = "l", xlab = "Epoch", ylab = "Beta Value", 
     main = "Semi Hessian Optimization lr_b = 0.1",
     sub = "Green colr dashed lined need know of the DGP",
     ylim=c(-1,11))
lines(rep(5, epochs), col = "red")
lines(betas_31, type = "l", xlab = "Epoch", ylab = "Beta Value", main = "Beta Values vs. Epoch")
lines(rep(9, epochs), col = "red")
lines(betas_32, type = "l", xlab = "Epoch", ylab = "Beta Value", main = "Beta Values vs. Epoch")
lines(rep(0.3, epochs), col = "red")

# Comparing with Colr
df = as.data.frame(train$df_orig$numpy())
colnames(df) = c('X1', 'X2', 'X3')
df$X4 = sqrt(train_labels[1:n_obs]) #GT not accessible in practivce
fit.colr = Colr(X3 ~ X1 + X2 + X4,df) #Note: X4 is not accessible in practice
confint(fit.colr)
dfp = as.numeric(confint(fit.colr)[1:2,])
for (i in 1:4) abline(h=dfp[i], col='green', lty=2)

dfp = confint(Colr(X2 ~ X1,df))
for (i in 1:4) abline(h=dfp[i], col='green')

# Save model name it fn+hessian+epoch
final_model$save_weights(paste0(fn, '_hessian_', epochs, '.h5'))
save.image(paste0(fn, '_hessian_', epochs, '.RData'))

} #if (FALSE) 
####### End of Hessian 

####### Training with Warm start #########
if (FALSE){
  final_model$get_layer(name = "beta")$get_weights()
weights = tf$constant(
  as.matrix(c(
   0.,5. ,9.0,
   0.,0.,0.3,
   0.,0.,0.
  ), nrow = 3), shape = c(3L,3L))
final_model$get_layer(name = "beta")$set_weights(list(weights))
final_model$get_layer(name = "beta")$get_weights()[[1]]

  ws = data.frame(w12=0.5, w13=0.2, w23=0.03)
  for (e in 1:100){
    print(e)
    final_model$fit(x = list(train$df_orig, train_images), 
                           y=train$df_orig, 
                           epochs = 1L,verbose = TRUE)
    w = final_model$get_layer(name = "beta")$get_weights()[[1]]
    #final_model$get_layer(name = "beta")$set_weights(list(weights))
    ws = rbind(ws, c(w[1,2], w[1,3], w[2,3]))  
  }
  plot(0:200, ws[,1], type='l', ylim=c(-1,1))
  lines(0:200, ws[,2], col='red')
  lines(0:200, ws[,3], col='blue')
  abline(h=0.5, col='black', lty=2)
  abline(h=0.2, col='red', lty=2)
  abline(h=0.03, col='blue', )
  legend("topright", legend = c("w12 0.5", "w13 0.2", "w23 0.03"), col = c("black", "red", "blue"), lty = 1:1, cex = 0.8)
} #End of Warmstart

## This function freezes all weights except the layer called 'beta'
freeze_weights = function(model){
  layers = model$layers
  for (i in 1:length(layers)){
    #if (layers[[i]]$name != 'beta'){
    if (i < 18){
      layers[[i]]$trainable = FALSE
    } else {
      layers[[i]]$trainable = TRUE
    }
  }
  #return(model)
}

release_weights = function(model){
  layers = model$layers
  for (i in 1:length(layers)){
    layers[[i]]$trainable = TRUE
  }
  #return(model)
}

print(final_model)
freeze_weights(final_model)
print(final_model)
release_weights(final_model)
print(final_model)

# Optimizers for different phases
#optimizer_frozen <- optimizer_adam(learning_rate = 1e-3)  # Larger learning rate for frozen phase
#optimizer_unfrozen <- optimizer_adam(learning_rate = 1e-3)  # Smaller learning rate for unfrozen phase


####### Normal Training ####
final_model
if (file.exists(fn)){
  final_model$load_weights(fn)
} else {
  # Initialize a data frame to store specific weights
  ws <- data.frame(w12 = numeric(), w13 = numeric(), w23 = numeric())
  # Initialize lists to store loss history
  train_loss <- numeric()
  val_loss <- numeric()
  
  # Training loop
  num_epochs <- 100
  for (e in 1:num_epochs) {
    print(paste("Epoch", e))
    # if (e < Inf) {
    #   # Release weights and compile with the smaller learning rate
    #   release_weights(final_model)
    #   final_model$compile(optimizer = optimizer_unfrozen, loss = semi_struct_dag_loss)
    # } else {
    #   # Freeze weights and compile with the larger learning rate
    #   freeze_weights(final_model)
    #   final_model$compile(optimizer = optimizer_frozen, loss = semi_struct_dag_loss)
    # }
    
    
    hist <- final_model$fit(x = list(train$df_orig, train_images), 
                            y = train$df_orig, 
                            epochs = 1L, verbose = TRUE, 
                            validation_data = list(list(test$df_orig, test_images), test$df_orig))
    
    # Append losses to history
    train_loss <- c(train_loss, hist$history$loss)
    val_loss <- c(val_loss, hist$history$val_loss)
    
    # Extract specific weights
    w <- final_model$get_layer(name = "beta")$get_weights()[[1]]
    ws <- rbind(ws, data.frame(w12 = w[1, 2], w13 = w[1, 3], w23 = w[2, 3]))
  }
  
  final_model$save_weights(paste0(fn, '_normal_', num_epochs, '.h5'))
  save.image(paste0(fn, '_normal_', num_epochs, '.RData'))
  #pdf(paste0('loss_',fn,'.pdf'))
  epochs = length(train_loss)
  plot(1:length(train_loss), train_loss, type='l', main='Normal Training', ylim=c(-0.5, -0.4))
  lines(1:length(train_loss), val_loss, type = 'l', col = 'green')
  
  plot(1:epochs, ws[,1], type='l', ylim=c(0,10), main='Normal Training')
  lines(1:epochs, ws[,2], col='black')
  lines(1:epochs, ws[,3], col='black')
  abline(h=9, col='red')
  abline(h=5, col='red')
  abline(h=0.3, col='red', )
  legend("topleft", legend = c("w12 5", "w13 2", "w23 0.3"), col = c("black", "red", "blue"), lty = 1:1, cex = 0.8)

  
  #Save a pdf of the loss function containing fn using pdf()
  #dev.off()
}

# final_model$evaluate(x = list(train$df_scaled, train_images), 
#                      y=train$df_scaled, batch_size = 7L)
fn
len_theta
final_model$get_layer(name = "beta")$get_weights() * final_model$get_layer(name = "beta")$mask
param_model = final_model



DER CODE WEITER UNTEN IST NICHT GETESTET 


# Check the derivatives of h w.r.t. x
x <- tf$ones(shape = c(10L, 3L)) #B,P
with(tf$GradientTape(persistent = TRUE) %as% tape, {
  tape$watch(x)
  y <- param_model(x)
})
d <- tape$jacobian(y, x)
for (k in 1:(2+len_theta)){ #k = 1
  print(k) #B,P,k,B,P
  B = 1
  print(d[B,,k,B,]) #
}

h_params = final_model(list(train$df_orig, train_images))
h_params

### Checking
x1 = train$df_orig$numpy()[1:1000,1]
b = train_labels[1:1000]
lm(b ~ x1)
plot(x1 , b)
abline(lm(b ~ x1))
plot(table(x1, b), col=TRUE)

eta_hat = h_params[1:1000,3,9]$numpy() 
eta_true = sqrt(train_labels[1:1000])
plot(eta_true, eta_hat)
confint(lm(eta_hat ~ eta_true))
abline(lm(eta_hat ~ eta_true))
cor(eta_hat, eta_true)

h_params[1:1000,3,1]$numpy()#CS 0
h_params[1:1000,3,2]$numpy()#CS 0


# Sampling fitted model w/o intervention --> OK 
s = do_dag_struct(param_model, train$A, doX=c(NA, NA, NA), num_samples = 5000)
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  hist(d, freq=FALSE, 20, xlim=c(0.1,1.1), main=paste0("X_",i))
  lines(density(train$df_scaled$numpy()[,i]))
}

######### Simulation of do-interventions #####
doX=c(0.5, NA, NA)
dx5 = dgp(10000, doX=doX)
dx5$df_orig$numpy()[1:5,]
dx5$df_scaled$numpy()[1:5,]
s_0.5 = scaled_doX(doX, train)[1]$numpy()


doX=c(0.7, NA, NA)
dx7 = dgp(10000, doX=doX)
hist(dx5$df_orig$numpy()[,2], freq=FALSE,100)
mean(dx7$df_orig$numpy()[,2]) - mean(dx5$df_orig$numpy()[,2])  
mean(dx7$df_orig$numpy()[,3]) - mean(dx5$df_orig$numpy()[,3])  
s_0.7 = scaled_doX(doX, train)[1]$numpy()

########### Do(x1) seems to work#####
s = do_dag_struct(param_model, train$A, doX=c(s_0.5, NA, NA))
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  print(mean(d))
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("Do (X3=0.5) X_",i))
  #lines(density(train$df_scaled$numpy()[,i]))
}
sdown = unscale(train$df_orig, s)
hist(sdown$numpy()[,3], freq=FALSE)
median(sdown$numpy()[,3])

s = do_dag_struct(param_model, train$A, doX=c(s_0.7, NA, NA))
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  print(mean(d))
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("Do (X3=0.7) X_",i))
  #lines(density(train$df_scaled$numpy()[,i]))
}
sup = unscale(train$df_orig, s)
hist(sup$numpy()[,3], freq=FALSE)
mean(sup$numpy()[,3])

mean(sup$numpy()[,2]) - mean(sdown$numpy()[,2])
mean(sup$numpy()[,3]) - mean(sdown$numpy()[,3])

median(sup$numpy()[,2]) - median(sdown$numpy()[,2])
median(sup$numpy()[,3]) - median(sdown$numpy()[,3])


####################################################################################
################ Below not tested after refactoring to new DGPs (Using Tranformation models in DGP) 
############################
####################################################################################

########### Do(x2) seem to work #####
s = do_dag_struct(param_model, train$A, doX=c(NA, 0.5, NA))
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  print(mean(d))
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("Do (X3=0.5) X_",i))
  #lines(density(train$df_scaled$numpy()[,i]))
}

s = do_dag_struct(param_model, train$A, doX=c(NA, 0.7, NA))
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  print(mean(d))
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("Do (X3=0.5) X_",i))
  #lines(density(train$df_scaled$numpy()[,i]))
}



########### TODO Check the sampling (prob needs ad) #####

dox1_2=scale_value(train$df_orig, col=1L, 2) #On X2
s_dox1_2 = do_dag(param_model, train$A, doX=c(dox1_2$numpy(), NA, NA), num_samples = 5000)
s = s_dox1_2
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("Do (X3=0.5) X_",i))
  lines(density(train$df_scaled$numpy()[,i]))
}
mean(s_dox1_2$numpy()[,3])
df = unscale(train$df_orig, s_dox1_2)
mean(df$numpy()[,3]) #1.39

dox1_3=scale_value(train$df_orig, col=1L, 3.) #On X2
s_dox1_3 = do_dag(param_model, train$A, doX=c(dox1_3$numpy(), NA, NA), num_samples = 5000)
mean(s_dox1_3$numpy()[,3])
df = unscale(train$df_orig, s_dox1_3)
mean(df$numpy()[,3]) #2.12

dox1_3=scale_value(train$df_orig, col=1L, 1.) #On X2
s_dox1_3 = do_dag(param_model, train$A, doX=c(dox1_3$numpy(), NA, NA), num_samples = 5000)
mean(s_dox1_3$numpy()[,3])
df = unscale(train$df_orig, s_dox1_3)
mean(df$numpy()[,3]) #0.63
t.test(df$numpy()[,3])


hist(train$df_scaled$numpy()[,1], freq=FALSE)

if(FALSE){
  x = tf$ones(c(2L,3L)) * 0.5
  # Define the MLP model
  input_layer <- layer_input(shape = list(ncol(adjacency)))
  d = layer_dense(units = 64, activation = 'relu')(input_layer)
  d = layer_dense(units = 30)(d)
  d = layer_reshape(target_shape = c(3, 10))(d)
  param_model = keras_model(inputs = input_layer, outputs = d)
  print(param_model)
  param_model(x)
  tf$executing_eagerly()  # Should return TRUE
  with(tf$GradientTape(persistent = TRUE) %as% tape, {
    theta_tilde = param_model(x, training=TRUE)
    loss = dag_loss(x, theta_tilde)
  })
  #gradients <- lapply(gradients, function(g) tf$debugging$check_numerics(g, "Gradient NaN/Inf check"))
  gradients = tape$gradient(loss, param_model$trainable_variables)
  gradients
  
  param_model$trainable_variables
  # Update weights
  optimizer.apply_gradients(zip(gradients, param_model.trainable_variables))
}




















