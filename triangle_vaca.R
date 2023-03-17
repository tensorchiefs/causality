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

forward_pass = function(X) {
  parz = triangle_enc(X,m1 = m1, a1 = a1, u1 = u1)
  #Std of latent variable
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
  
  
  #kl_loss_1 = -0.5 * tf$reduce_sum(1. + z1s - tf$square(z1mu) - tf$exp(z1s), 1L)
  #kl_loss_2 = -0.5 * tf$reduce_sum(1. + z2s - tf$square(z2mu) - tf$exp(z2s), 1L)
  #kl_loss_3 = -0.5 * tf$reduce_sum(1. + z3s - tf$square(z3mu) - tf$exp(z3s), 1L)
  
  #The loss is -ELBO = nELBO = NLL + DKL 
  #THE DKL 
  kl_loss_1 = -0.5 * tf$reduce_sum(1. + tf$math$log(tf$square(z1s)) - tf$square(z1mu) - tf$square(z1s), 1L)
  kl_loss_2 = -0.5 * tf$reduce_sum(1. + tf$math$log(tf$square(z2s)) - tf$square(z2mu) - tf$square(z2s), 1L)
  kl_loss_3 = -0.5 * tf$reduce_sum(1. + tf$math$log(tf$square(z3s)) - tf$square(z3mu) - tf$square(z3s), 1L)
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
  
  
  nll_loss_1 = tf$reduce_sum(tf$math$log(o1_s) + (tf$square(X[,1] - o1_mu)/(2.0 * tf$square(o1_s))))
  nll_loss_2 = tf$reduce_sum(tf$math$log(o2_s) + (tf$square(X[,2] - o2_mu)/(2.0 * tf$square(o2_s))))
  nll_loss_3 = tf$reduce_sum(tf$math$log(o3_s) + (tf$square(X[,3] - o3_mu)/(2.0 * tf$square(o3_s))))
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

loss = forward_pass(X)
# Use ADAM optimizer
optimizer =  tf$optimizers$Adam()

##### Collecting all weights



train_step = function(){
  ### --- TODO move out of function and check
  tvars = list(m1$trainable_variables,   #message NN encoder  Input --> Z
               u1$trainable_variables,   #update NN encoder 
               m1z$trainable_variables,  #message NN decoder z-->h
               u1z$trainable_variables,  #update NN decoder z-->h
               mho$trainable_variables,  #message NN decoder h-->output
               uho$trainable_variables)  #message NN decoder h-->output
  
  with(tf$GradientTape() %as% tape, {
    loss = forward_pass(X)
    grads = tape$gradient(loss, tvars)
    for (i in 1:length(tvars)){
      optimizer$apply_gradients(
        purrr::transpose(list(grads[[i]], tvars[[i]]))
      )  
    }
  }
  )
  return (loss)
}

train_step_faster = tf_function(train_step)
T_STEP = 15000
losses = rep(NA, T_STEP)
for (r in 1:T_STEP){
  losses[r] = train_step_faster()$numpy()  
  #losses[r] = train_step()$numpy()    
  if ((r %% 10) == 0) {
    print(paste(r, losses[r]))
  }
}

plot(1:T_STEP, losses, ylim=c(5000, 8000))

###### Sampling from the decoder 
BSS = 5000L
Z1 = tf$random$normal(c(BSS, 1L))
Z2 = tf$random$normal(c(BSS, 1L))
Z3 = tf$random$normal(c(BSS, 1L))

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

X1s = rnorm(n = BSS, mean = o1_mu$numpy(), sd=o1_s$numpy())
X2s = rnorm(n = BSS, mean = o2_mu$numpy(), sd=o2_s$numpy())
X3s = rnorm(n = BSS, mean = o3_mu$numpy(), sd=o3_s$numpy())


Xobs = X$numpy()
hist(X1s,100, freq = FALSE)
lines(density(Xobs[,1]),col='red')

hist(X2s,100, freq = FALSE)
lines(density(Xobs[,2]),col='red')

hist(X3s,100, freq = FALSE)
lines(density(Xobs[,3]),col='red')

###### Learnend Encoder ######
parz = triangle_enc(X, m1, a1, u1)
z1s = tf$math$softplus(parz[[1]][,2,drop=FALSE])$numpy()
z2s = tf$math$softplus(parz[[2]][,2,drop=FALSE])$numpy()
z3s = tf$math$softplus(parz[[3]][,2,drop=FALSE])$numpy()

#Expecation of latent variable
z1mu = parz[[1]][,1,drop=FALSE]$numpy()
z2mu = parz[[2]][,1,drop=FALSE]$numpy()
z3mu = parz[[3]][,1,drop=FALSE]$numpy()

Z1s = rnorm(n = BS, mean = z1mu, sd=z1s)
hist(z1s,1000)

hist(z1mu, 100)

hist(o1_mu$numpy(),100)
