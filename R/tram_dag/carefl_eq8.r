source('R/tram_dag/utils.R')
library(R.utils)
DEBUG = FALSE
DEBUG_NO_EXTRA = FALSE
SUFFIX = 'run500'
DROPBOX = 'C:/Users/sick/dl Dropbox/beate sick/IDP_Projekte/DL_Projekte/shared_Oliver_Beate/Causality_2022/tram_DAG/'
DROPBOX = '~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/Causality_2022/tram_DAG/'
if (rstudioapi::isAvailable()) {
  context <- rstudioapi::getSourceEditorContext()
  this_file <- context$path
  print(this_file)
}

#######################
# Latent Distribution
latent_dist = tfd_logistic(loc=0, scale=1)
#latent_dist = tfd_normal(loc=0, scale=1)
#latent_dist = tfd_truncated_normal(loc=0., scale=1.,low=-4,high = 4)
hist(latent_dist$sample(1e5)$numpy(),100, freq = FALSE, main='Samples from Latent')

M = 30
len_theta = M + 1
bp = make_bernp(len_theta)

######################################
############# DGP ###############
######################################
#coeffs <- runif(2, min = .1, max = .9)
coeffs = c(.3,.7)

# https://github.com/piomonti/carefl/blob/master/data/generate_synth_data.py

rlaplace <- function(n, location = 0, scale = 1) {
  p <- runif(n) - 0.5
  draws <- location - scale * sign(p) * log(1 - 2 * abs(p))
  return(draws)
}

hist(rlaplace(10000),100)
sd(rlaplace(10000))

dgp <- function(n_obs, coeffs, doX1=NA, dat_train=NULL) {
  #X <- matrix(rlaplace(2 * n_obs, 0, 1 / sqrt(2)), nrow = 2, ncol = n_obs)
  X <- matrix(rnorm(2 * n_obs, 0, 1), nrow = 2, ncol = n_obs)
  X_1 <- X[1,]
  X_2 <- X[2,]
  
  if (is.na(doX1) == FALSE){
    X_1 = X_1 * 0 + doX1
  }
  
  X_3 <- X_1 + coeffs[1] * (X_2^3)
  X_4 <- -X_2 + coeffs[2] * (X_1^2)
  #X_3 <- X_3 + rlaplace(n_obs, 0, 1 / sqrt(2))
  #X_4 <- X_4 + rlaplace(n_obs, 0, 1 / sqrt(2))
  
  X_3 <- X_3 + rnorm(n_obs, 0, 1 )
  X_4 <- X_4 + rnorm(n_obs, 0, 1 )
  
  dat.s =  data.frame(x1 = X_1, x2 = X_2, x3 = X_3, x4 = X_4)
  dat.tf = tf$constant(as.matrix(dat.s), dtype = 'float32')
  A <- matrix(c(0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,0,0), nrow = 4, ncol = 4, byrow = TRUE)
  
  if (is.null(dat_train)){
    scaled = scale_df(dat.tf)
  } else{
    scaled = scale_validation(dat_train, dat.tf)
  }
  return(list(df_orig=dat.tf, df_scaled = scaled, coef=coeffs, A=A, name='carefl_eq8'))
} 

train = dgp(1000, coeffs = coeffs)
pairs(train$df_orig$numpy())
pairs(train$df_scaled$numpy())
train$coef
train$A
library(igraph)
graph <- graph_from_adjacency_matrix(train$A, mode = "directed", diag = FALSE)
plot(graph, vertex.color = "lightblue", vertex.size = 30, edge.arrow.size = 0.5)

net = make_net(train$A, train$df_scaled, len_theta = len_theta)

val = dgp(5000, coeffs = coeffs, dat_train = train$df_orig)
net_val = make_net(val$A, val$df_scaled, len_theta = len_theta)
#We don;t nee the network
net_val$thetaNN = NULL


###### Training Step #####
dirname = paste0(DROPBOX, "exp/", train$name, "/", SUFFIX, "/")
file.copy(this_file, dirname)

optimizer= tf$keras$optimizers$Adam(learning_rate=0.001)
l = do_training(train$name, net=net, net_val=net_val,SUFFIX, epochs = 200,  optimizer=optimizer)
#e:200.000000  Train: -2.883684, Val: -2.179241 
loss = l[[1]]
loss_val = l[[2]]
plot(loss, type='l', ylim=c(-4.0,5))
points(loss_val, col='green')


#######################
# Below ad-hoc evaluation
thetaNN_l = net$thetaNN
parents_l = net$parents
target_l = net$target

parents_l_val = net_val$parents
target_l_val = net_val$target
#Loading data from epoch e
e = 200
for (i in 1:ncol(val$A)){
  fn = paste0(dirname,train$name, "_nn", i, "_e", e, "_weights.h5")
  thetaNN_l[[i]]$load_weights(path.expand(fn))
}

val2 = dgp(10000, coeffs = coeffs, dat_train = train$df_orig)
net_val2 = make_net(val2$A, val2$df_scaled, len_theta = len_theta)
parents_l_val2 = net_val2$parents
target_l_val2 = net_val2$target
net_val2$thetaNN = NULL
  
NLL_val = NLL_train = NLL_val2 = 0  
for(i in 1:ncol(val$A)) { # Assuming that thetaNN_l, parents_l and target_l have the same length
  NLL_train = NLL_train + calc_NLL(thetaNN_l[[i]], parents_l[[i]], target_l[[i]])$numpy()
  NLL_val = NLL_val + calc_NLL(thetaNN_l[[i]], parents_l_val[[i]], target_l_val[[i]])$numpy()
  NLL_val2 = NLL_val2 + calc_NLL(thetaNN_l[[i]], parents_l_val2[[i]], target_l_val2[[i]])$numpy()
}
NLL_val
loss_val[(length(loss_val)-10):length(loss_val)]
loss[(length(loss)-10):length(loss)]
NLL_train
NLL_val2

##########. Checking obs fit the marginals 
plot_obs_fit(parents_l, target_l, thetaNN_l, name='Training')
plot_obs_fit(parents_l_val, target_l_val, thetaNN_l, name='Validation')
plot_obs_fit(parents_l_val2, target_l_val2, thetaNN_l, name='Validation 2')

############################### Do X via Flow ########################
#Samples from Z give X=doX
dox1_eq8 = function(doX, thetaNN_l, num_samples){
  doX_tensor = doX * tf$ones(shape=c(num_samples,1L),dtype=tf$float32) 
  
  parents_x2 = tf$ones(shape=c(num_samples,1L),dtype=tf$float32) #No parents --> set to one (see parents_tmp)
  x2_samples = sample_from_target(thetaNN_l[[2]], parents_x2)
  
  parents_x3 = tf$concat(list(doX_tensor, x2_samples), axis=1L)
  x3_samples = sample_from_target(thetaNN_l[[3]], parents_x3)
  
  parents_x4 = tf$concat(list(doX_tensor, x2_samples), axis=1L)
  x4_samples = sample_from_target(thetaNN_l[[4]], parents_x4)
  
  return(matrix(c(doX_tensor$numpy(),x2_samples$numpy(), x3_samples$numpy(), x4_samples$numpy()), ncol=4))
}


dox_origs = seq(-3, 3, by = 0.2)
res_scm = res = dox_origs
for (i in 1:length(dox_origs)){
  dox_orig = dox_origs[i]
  #dox_orig = -2 # we expect E(X3|X1=dox_orig)=dox_orig
  dox=scale_value(train$df_orig, col=1L, dox_orig)
  num_samples = 1000L
  dat_do_x_s = dox1_eq8(dox, thetaNN_l, num_samples = num_samples)
  mean(dat_do_x_s[,3])
  df = unscale(train$df_orig, dat_do_x_s)
  res[i] = mean(df[,3]$numpy())
  
  d = dgp(1000L, coeffs = coeffs, doX1=dox_orig)
  res_scm[i] = mean(d$df_orig[,3]$numpy())
}
mean((res - dox_origs)^2)
plot(dox_origs, res)
abline(0,1)
points(dox_origs, res_scm, col='green')


hist(train$df_orig[,1]$numpy(),100)
abline(v=0)

