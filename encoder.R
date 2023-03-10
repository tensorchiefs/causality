options(drop = FALSE)
library(keras)
library(tensorflow)

dim_x = 1
dim_z = 1
dim_h = 1

m1 = keras_model_sequential() %>% 
  layer_dense(units =  5, activation = 'relu', input_shape = c(2*dim_x)) %>% 
  layer_dense(units =  dim_x)
  
a1 = keras$layers$average

u1 = keras_model_sequential() %>% 
  layer_dense(units =  5, activation = 'relu', input_shape = c(2*dim_x)) %>% 
  layer_dense(units =  2L*dim_z)

summary(u1)

if(FALSE){
  u1_1 <- layer_input(shape = c(1))
  u1_2 <- layer_input(shape = c(1))
  d = layer_concatenate(list(u1_1, u1_2)) %>% 
    layer_dense(units =  5, activation = 'relu') %>% 
    layer_dense(units =  2L*dim_z)
  u1 <- keras_model(inputs = list(u1_1, u1_2), outputs = d)
}

summary(u1)

triangle_enc = function(X, m1, a1, u1){
  #X[N,3] for the 3 variables X1,X2,X3
  #m1 is NN for msg in layer 1
  #a1 parameter free aggregation function
  #u1 is NN for updates in layer 1
  #X = matrix(rnorm(30), ncol=3)
  X11 = tf$`repeat`(X, repeats=c(2L,0L,0L), axis=1L)
  X22 = tf$`repeat`(X, repeats=c(0L,2L,0L), axis=1L)
  X33 = tf$`repeat`(X, repeats=c(0L,0L,2L), axis=1L)
  
  m11 = m1(X11)
  m22 = m1(X22)
  m33 = m1(X33)
  
  m32 = m1(X[,c(2L,3L)])
  m31 = m1(X[,c(1L,3L)])
  m21 = m1(X[,c(1L,2L)])
  
  M1 = a1(list(m11))
  M2 = a1(list(m21, m22))
  M3 = a1(list(m31, m32, m33))
  
  z1_mu_sigma = u1(tf$concat(list(X[,1L,drop=FALSE], M1), axis=1L))
  z2_mu_sigma = u1(tf$concat(list(X[,2L,drop=FALSE], M2), axis=1L))
  z3_mu_sigma = u1(tf$concat(list(X[,3L,drop=FALSE], M3), axis=1L))
  
  return(list(z1_mu_sigma,z2_mu_sigma,z3_mu_sigma))
}  

###------ Z to Hidden ####
m1z = keras_model_sequential() %>% 
  layer_dense(units =  5, activation = 'relu', input_shape = c(2*dim_z)) %>% 
  layer_dense(units =  dim_h)

#TODO check output dimension dim_h
u1z = keras_model_sequential() %>% 
  layer_dense(units =  5, activation = 'relu', input_shape = c(2*dim_z)) %>% 
  layer_dense(units =  dim_h)

###------ Hidden to output ####
mho = keras_model_sequential() %>% 
  layer_dense(units =  5, activation = 'relu', input_shape = c(2*dim_h)) %>% 
  layer_dense(units =  dim_h)

uho = keras_model_sequential() %>% 
  layer_dense(units =  5, activation = 'relu', input_shape = c(2*dim_h)) %>% 
  layer_dense(units =  2*dim_x)

triangle_dec = function(Z1,Z2,Z3, m1z, u1z, mho, uho){
  Z11 = tf$concat(list(Z1,Z1), axis=1L)
  Z22 = tf$concat(list(Z2,Z2), axis=1L)
  Z33 = tf$concat(list(Z3,Z3), axis=1L)
  
  m11 = m1z(Z11)
  m22 = m1z(Z22)
  m33 = m1z(Z33)
  
  m32 = m1z(tf$concat(list(Z3,Z2), axis=1L))
  m31 = m1z(tf$concat(list(Z3,Z1), axis=1L))
  m21 = m1z(tf$concat(list(Z2,Z1), axis=1L)) 
  
  M1 = a1(list(m11))
  M2 = a1(list(m21, m22))
  M3 = a1(list(m31, m32, m33))
    
  h1 = u1z(tf$concat(list(Z1, M1), axis=1L))
  h2 = u1z(tf$concat(list(Z2, M2), axis=1L))
  h3 = u1z(tf$concat(list(Z3, M3), axis=1L))
  
  ############## Hidden --> Output #######################
  h11 = tf$concat(list(h1,h1), axis=1L)
  h22 = tf$concat(list(h2,h2), axis=1L)
  h33 = tf$concat(list(h3,h3), axis=1L)
  
  m11 = mho(h11)
  m22 = mho(h22)
  m33 = mho(h33)
  
  m32 = m1z(tf$concat(list(h3,h2), axis=1L))
  m31 = m1z(tf$concat(list(h3,h1), axis=1L))
  m21 = m1z(tf$concat(list(h2,h1), axis=1L)) 
  
  M1 = a1(list(m11))
  M2 = a1(list(m21, m22))
  M3 = a1(list(m31, m32, m33))
  
  o1_mu_sigma = uho(tf$concat(list(h1, M1), axis=1L))
  o2_mu_sigma = u1z(tf$concat(list(h2, M2), axis=1L))
  o3_mu_sigma = u1z(tf$concat(list(h3, M3), axis=1L))  
  
  return(list(o1_mu_sigma,o2_mu_sigma,o3_mu_sigma))
}

train_step = function(X) {
  parz = triangle_enc(X,m1 = m1, a1 = a1, u1 = u1)
  #Variance of latent variable
  z1s = tf$math$softplus(parz[[1]][,2,drop=FALSE])
  z2s = tf$math$softplus(parz[[2]][,2,drop=FALSE])
  z3s = tf$math$softplus(parz[[3]][,2,drop=FALSE])
  
  #Expecation of latent variable
  z1mu = parz[[1]][,1,drop=FALSE]
  z2mu = parz[[2]][,1,drop=FALSE]
  z3mu = parz[[3]][,1,drop=FALSE]
  
  Z1 = z1mu +  z1s * tf$random$normal(c(BS,1L))
  Z2 = z2mu +  z2s * tf$random$normal(c(BS,1L))
  Z3 = z3mu +  z3s * tf$random$normal(c(BS,1L))
  
  
  kl_loss_1 = -0.5 * tf$reduce_sum(1. + z1s - tf$square(z1mu) - tf$exp(z1s), 1L)
  kl_loss_2 = -0.5 * tf$reduce_sum(1. + z2s - tf$square(z2mu) - tf$exp(z2s), 1L)
  kl_loss_3 = -0.5 * tf$reduce_sum(1. + z3s - tf$square(z3mu) - tf$exp(z3s), 1L)
  kl_loss = kl_loss_1 + kl_loss_2 + kl_loss_3
  
  d = triangle_dec(Z1,Z2,Z3, m1z, u1z, mho, uho)
  o1_mu_sigma = d[[1]]  
  o2_mu_sigma = d[[2]]
  o3_mu_sigma = d[[3]]
  
  o1_mu = o1_mu_sigma[,1]
  o2_mu = o2_mu_sigma[,1]
  o3_mu = o3_mu_sigma[,1]
  
  o1_s = tf$math$softplus(o1_mu_sigma[,1])
  o2_s = tf$math$softplus(o2_mu_sigma[,1])
  o3_s = tf$math$softplus(o3_mu_sigma[,1])
  
  
  nll_loss_1 = tf$reduce_sum(0.5 * o1_s + (tf$square(X[,1] - o1_mu)/(2.0 * tf$exp(o1_s))))
  nll_loss_2 = tf$reduce_sum(0.5 * o2_s + (tf$square(X[,1] - o2_mu)/(2.0 * tf$exp(o2_s))))
  nll_loss_3 = tf$reduce_sum(0.5 * o3_s + (tf$square(X[,1] - o3_mu)/(2.0 * tf$exp(o3_s))))
  nll_loss = nll_loss_1 + nll_loss_2 + nll_loss_3
  
  loss = tf$reduce_mean(nll_loss + kl_loss)   # average over batch
  return(loss)
}

################# Training Data
library(reticulate)
np <- import("numpy")
my_array <- np$load("/Users/oli/Documents/workspace_other/VACA/VACA/triangle_linear/train_5000_X.npy")
U <- np$load("/Users/oli/Documents/workspace_other/VACA/VACA/triangle_linear/train_5000_U.npy")
#X = matrix(rnorm(3L*BS), ncol=3)
BS = nrow(my_array)
X = k_constant(my_array)

loss = train_step(X)
# Use ADAM optimizer
optimizer =  tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
