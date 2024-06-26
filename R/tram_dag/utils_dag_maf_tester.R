library(keras)
library(tensorflow)
source('R/tram_dag/utils_dag_maf.R') #Might be called twice

###### Tesing the MAF ######
#Defines MAF
hidden_features <- c(2, 2)
order <- 4L
bs <- 5L
dim <- 3L
adjacency <- matrix(c(0, 1, 1, 0, 0, 0, 0, 0, 0), nrow = 3, byrow = FALSE)
adjacency <- adjacency == 1  # Convert to a boolean matrix

masks <- create_masks(adjacency, hidden_features)  # Custom function

layer_sizes <- c(ncol(adjacency), hidden_features, nrow(adjacency))
model <- create_theta_tilde_maf(adjacency, order, layer_sizes)  # Custom function

x <- tf$ones(shape = c(bs, dim))
theta_tilde <- model(x)
stopifnot(tf$shape(theta_tilde)$numpy() == c(bs, dim, order))  # Shape check

with(tf$GradientTape(persistent = TRUE) %as% tape, {
  tape$watch(x)
  y <- model(x)
})
d <- tape$jacobian(y, x)
stopifnot(tf$shape(d)$numpy() == c(bs, dim, order, bs, dim))  # Shape check
print(d)


source('R/tram_dag/utils.R') #L_START
### Testing h and h_dash #######
if (FALSE){
  M = 4L #The order of the bernstein polynoms
  len_theta = M + 1L
  t_i = tf$Variable(matrix(rep(seq(-1,2,0.5),3), ncol=3))#
  t_i = tf$Variable(matrix(rep(seq(-1,2,0.01),3), ncol=3))
  t_i = tf$cast(t_i, dtype=tf$float32)
  
  theta_val = matrix(rep(c(0.,2.,2.,2.,2.),3), byrow = TRUE, ncol=len_theta)
  batch_size = 7L
  batch_size = 301L
  theta_batch = tf$tile(tf$expand_dims(theta_val, 0L), c(batch_size, 1L, 1L))
  theta = tf$cast(theta_batch, dtype=tf$float32)
  h_dag_dashdd = h_dag_dash(t_i, theta)
  h_ti = h_dag_extra(t_i, theta)
  h = h_dag(t_i, theta)
  h_dag_dash_extrad= h_dag_dash_extra(t_i, theta)
  plot(t_i$numpy()[,1],h_ti$numpy()[,1], type='l')
  lines(t_i$numpy()[,1],h$numpy()[,1], type='l', col='red')
  
  plot(t_i$numpy()[,1],h_dag_dashdd$numpy()[,1], type='l')
  lines(t_i$numpy()[,1],h_dag_dash_extrad$numpy()[,1], type='l', col='red')
  
  dag_loss(t_i, theta)
  
  #lines(t_i$numpy()[,1],h_dag_dashd$numpy()[,1], col='red')
}

##### TEMP
dgp <- function(n_obs) {
    print("=== Using the DGP of the VACA1 paper in the linear Fashion (Tables 5/6)")
    flip = sample(c(0,1), n_obs, replace = TRUE)
    X_1 = flip*rnorm(n_obs, -2, sqrt(1.5)) + (1-flip) * rnorm(n_obs, 1.5, 1)
    X_2 = -X_1 + rnorm(n_obs)
    X_3 = X_1 + 0.25 * X_2 + rnorm(n_obs)
    dat.s =  data.frame(x1 = X_1, x2 = X_2, x3 = X_3)
    dat.tf = tf$constant(as.matrix(dat.s), dtype = 'float32')
    scaled = scale_df(dat.tf) * 0.99 + 0.005
    
    A <- matrix(c(0, 1, 1, 0,0,1,0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
    return(list(df_orig=dat.tf,  df_scaled = scaled, A=A))
} 

train = dgp(500)
hist(train$df_scaled$numpy()[,1],50)
hist(train$df_scaled$numpy()[,2],50)
hist(train$df_scaled$numpy()[,3],50)


hist(train$df_orig$numpy()[,1],50)
hist(train$df_orig$numpy()[,2],50)
hist(train$df_orig$numpy()[,3],50)
summary(train$df_orig$numpy())
  
library(igraph)
graph <- graph_from_adjacency_matrix(train$A, mode = "directed", diag = FALSE)
plot(graph, vertex.color = "lightblue", vertex.size = 30, edge.arrow.size = 0.5)
hidden_features = c(2,2)
adjacency <- t(train$A)
layer_sizes <- c(ncol(adjacency), hidden_features, nrow(adjacency))
#order = 5L

# Create Masks
masks = create_masks(adjacency = adjacency, hidden_features)
dag_maf_plot(masks, layer_sizes)


#source('tram_scm/model_utils.R')
M = 30L
param_model = create_theta_tilde_maf(adjacency = adjacency, len_theta = M+1, layer_sizes = layer_sizes)
param_model(train$df_scaled)
optimizer = optimizer_adam()
param_model$compile(optimizer, loss=dag_loss)
param_model$evaluate(x = train$df_scaled, y=train$df_scaled, batch_size = 3L)

##### Training ####
hist = param_model$fit(x = train$df_scaled, y=train$df_scaled, epochs = 2L,verbose = TRUE)
plot(hist$epoch, hist$history$loss)
if (FALSE){
  #register_keras_serializable(name = "dag_loss", namespace = "custom_namespace")
  #param_model$save('triangle_test.keras')
  param_model$save_weights('triangle_test_weights.h5')
  param_model$load_weights('triangle_test_weights.h5')
} 


s = do_dag(param_model, train$A, doX=c(NA, NA, NA))
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("X_",i))
  lines(density(train$df_scaled$numpy()[,i]))
}

s = do_dag(param_model, train$A, doX=c(0.5, NA, NA))
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("Do (X3=0.5) X_",i))
  lines(density(train$df_scaled$numpy()[,i]))
}

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




















