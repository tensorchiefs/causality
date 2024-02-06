library(keras)
library(tensorflow)
source('R/tram_dag/utils_dag_maf.R')

###### Tesing the MAF ######
#Defines MAF
hidden_features = c(2,2)
adjacency <- matrix(c(0, 1, 1, 0, 0, 0, 0, 0, 0), nrow = 3, byrow = FALSE)

# Create Masks
masks = create_masks(adjacency = adjacency, hidden_features)
layer_sizes <- c(ncol(adjacency), hidden_features, nrow(adjacency))


###### Testing a Linear Masked Layer
mask = masks[[1]]
l = LinearMasked(2, t(mask))
x = tf$ones(c(5L,3L))
l(x)

dag_maf_plot(masks, layer_sizes)
# Create MAF
param_model = create_theta_tilde_maf(adjacency = adjacency, 
                                     len_theta = 5L,
                                     layer_sizes = layer_sizes)
x = tf$ones(c(2L,3L))
theta_tilde = param_model(x)
theta_tilde

# Calculate the Jacobian matrix using Python and TensorFlow
with(tf$GradientTape(persistent = TRUE) %as% tape, {
  tape$watch(x)
  y <- param_model(x)
})
d = tape$jacobian(y, x)
d[1,1:3,5,1,1:3] #It's a bit strange that the jacobimatrix also has the batch


source('R/tram_dag/utils.R') #L_START
### Testing h and h_dash #######
if (FALSE){
  M = 4L #The order of the bernstein polynoms
  len_theta = M + 1L
  t_i = tf$Variable(matrix(rep(seq(-1,2,0.01),3), ncol=3))
  t_i = tf$cast(t_i, dtype=tf$float32)
  
  theta = matrix(rep(c(0.,2.,2.,2.,2.),3), byrow = TRUE, ncol=len_theta)
  batch_size = 301L
  theta_batch = tf$tile(tf$expand_dims(theta, 0L), c(batch_size, 1L, 1L))
  theta = tf$cast(theta_batch, dtype=tf$float32)
  h_dag_dashd = h_dag_dash(t_i, theta)
  h_ti = h_dag_extra(t_i, theta)
  plot(t_i$numpy()[,1],h_ti$numpy()[,1], type='l')
  lines(t_i$numpy()[,1],h_dag_dashd$numpy()[,1], col='red')
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
    scaled = scale_df(dat.tf) * 0.99 + 0.001
    
    A <- matrix(c(0, 1, 1, 0,0,1,0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
    return(list(df_orig=dat.tf,  df_scaled = scaled, A=A))
} 

train = dgp(72)

library(igraph)
graph <- graph_from_adjacency_matrix(train$A, mode = "directed", diag = FALSE)
plot(graph, vertex.color = "lightblue", vertex.size = 30, edge.arrow.size = 0.5)
hidden_features = c(2)
adjacency <- t(train$A)
layer_sizes <- c(ncol(adjacency), hidden_features, nrow(adjacency))
#order = 5L

# Create Masks
masks = create_masks(adjacency = adjacency, hidden_features)
dag_maf_plot(masks, layer_sizes)


#source('tram_scm/model_utils.R')
M = 5L
param_model = create_theta_tilde_maf(adjacency = adjacency, len_theta = M+1, layer_sizes = layer_sizes)
param_model(train$df_scaled)
optimizer = optimizer_adam(learning_rate = 0.00001)
param_model$compile(optimizer, loss=dag_loss)
param_model$evaluate(x = train$df_scaled, y=train$df_orig, batch_size = 32L)
hist = param_model$fit(x = train$df_scaled, y=train$df_orig, epochs = 10L,verbose = TRUE)

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

















