source('R/tram_dag/utils.R')
library(R.utils)
DEBUG = FALSE
DEBUG_NO_EXTRA = FALSE

SUFFIX = 'runNormal_1e3'

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

dgp <- function(n_obs, coeffs, doX1=NA, dat_train=NULL, seed=NA) {
  if (is.na(seed) == FALSE){
    set.seed(seed)
  }
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

train = dgp(1000, coeffs = coeffs, seed=42)
pairs(train$df_orig$numpy())
pairs(train$df_scaled$numpy())
train$coef
train$A
library(igraph)
graph <- graph_from_adjacency_matrix(train$A, mode = "directed", diag = FALSE)
plot(graph, vertex.color = "lightblue", vertex.size = 30, edge.arrow.size = 0.5)

train_data = split_data(train$A, train$df_scaled)
thetaNN_l = make_thetaNN(train$A, train_data$parents)

val = dgp(5000, coeffs = coeffs, dat_train = train$df_orig)
val_data = split_data(val$A, val$df_scaled)


###### Training Step #####
optimizer= tf$keras$optimizers$Adam(learning_rate=0.001)
l = do_training(train$name, thetaNN_l = thetaNN_l, train_data = train_data, val_data = val_data,
                SUFFIX, epochs = 500,  optimizer=optimizer)

### Save Script
dirname = paste0(DROPBOX, "exp/", train$name, "/", SUFFIX, "/")
file.copy(this_file, dirname)

#e:200.000000  Train: -2.883684, Val: -2.179241 
loss = l[[1]]
loss_val = l[[2]]
plot(loss, type='l', ylim=c(-6.0,5))
points(loss_val, col='green')


#######################
# Below ad-hoc evaluation
print("----------- Weights before loading ----------- ")
thetaNN_l[[4]]$get_weights()[[2]]
#saveRDS(thetaNN_l[[1]]$get_weights(), '/tmp/dumm1.rds')
#rm(thetaNN_l)
#thetaNN_l[[1]]$set_weights(readRDS('/tmp/dumm1.rds'))

#Loading data from epoch e
e = 500
for (i in 1:ncol(val$A)){
  #fn = paste0(dirname,train$name, "_nn", i, "_e", e, "_weights.h5")
  #thetaNN_l[[i]]$load_weights(path.expand(fn))
  fn = paste0(dirname,train$name, "_nn", i, "_e", e, "_weights.rds")
  thetaNN_l[[i]]$set_weights(readRDS(fn))
  
  #fn = paste0(dirname,train$name, "_nn", i, "_e", e, "_model.h5")
  #thetaNN_l[[i]] = load_model_hdf5(fn)
  
  printf('Layer %d checksum: %s \n',i, calculate_checksum(thetaNN_l[[i]]$get_weights()))
}

NLL_val = NLL_train = NLL_val2 = 0  
for(i in 1:ncol(val$A)) { # Assuming that thetaNN_l, parents_l and target_l have the same length
  NLL_train = NLL_train + calc_NLL(thetaNN_l[[i]], train_data$parents[[i]], train_data$target[[i]])$numpy()
  NLL_val = NLL_val + calc_NLL(thetaNN_l[[i]], val_data$parents[[i]], val_data$target[[i]])$numpy()
  #NLL_val2 = NLL_val2 + calc_NLL(thetaNN_l[[i]], parents_l_val2[[i]], target_l_val2[[i]])$numpy()
}
loss[(length(loss)-10):length(loss)]
NLL_train

NLL_val 
loss_val[(length(loss_val)-10):length(loss_val)]

##########. Checking obs fit the marginals 
plot_obs_fit(train_data$parents, train_data$target, thetaNN_l, name='Training')
plot_obs_fit(val_data$parents, val_data$target, thetaNN_l, name='Validation')

#DEBUG_NO_EXTRA = TRUE
#dd  = sample_from_target(thetaNN_l[[4]], val_data$parents[[4]])
#hist(x_samples$numpy(),100)
#stripchart(x_samples$numpy(), method = 'jitter')
#stripchart(dd$numpy(), method = 'jitter')


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

##################
# Do Interventions on x1
dox_origs = seq(-3, 3, by = 0.5)
res_med_x4  = res_scm_x4 = res_scm_x3 = res_x3 = res_x4 = dox_origs
for (i in 1:length(dox_origs)){
  dox_orig = dox_origs[i]
  #dox_orig = -2 # we expect E(X3|X1=dox_orig)=dox_orig
  dox=scale_value(train$df_orig, col=1L, dox_orig)
  num_samples = 1142L
  dat_do_x_s = dox1_eq8(dox, thetaNN_l, num_samples = num_samples)
  
  #
  df = unscale(train$df_orig, dat_do_x_s)
  res_x3[i] = mean(df[,3]$numpy())
  res_x4[i] = mean(df[,4]$numpy())
  #res_med_x4[i] = median(df[,4]$numpy())
  
  
  d = dgp(1000L, coeffs = coeffs, doX1=dox_orig)
  res_scm_x3[i] = mean(d$df_orig[,3]$numpy())
  res_scm_x4[i] = mean(d$df_orig[,4]$numpy())
}
#X3
mse = mean((res_x3 - dox_origs)^2)
x1dat = data.frame(x=train$df_orig$numpy()[,1])
library(ggplot2)
data.frame(
  x = dox_origs,
  ours = res_x3,
  theoretical = dox_origs,
  Simulation_DGP = res_scm_x3
) %>% 
  pivot_longer(cols = 2:4) %>% 
  ggplot(aes(x=x, y=value, col=name, type=name)) + geom_point() +
  geom_abline(intercept = 0, slope = 1, col='skyblue') +
  xlab("do(X1)") +
  ylab("E(X3|do(X1)") +
  geom_rug(data=x1dat, aes(x=x), inherit.aes = FALSE, alpha=0.5) +
  ggtitle(paste0('MSE ours vs theoretical: ', round(mse,4)))

#X4
mse = mean((res_x4 - coeffs[2] * dox_origs^2)^2)
dox1s = seq(-3,3, length.out=100) 
dat_theo = data.frame(
  x = dox1s,
  y = coeffs[2] * dox1s^2
)

library(ggplot2)
data.frame(
  x = dox_origs,
  ours = res_x4,
  theoretical = coeffs[2] * dox_origs^2,
  Simulation_DGP = res_scm_x4
) %>% 
  pivot_longer(cols = 2:4) %>% 
  ggplot(aes(x=x, y=value, col=name, type=name)) + geom_point() +
  xlab("do(X1)") +
  ylab("E(X4|do(X1)") +
  geom_rug(data=x1dat, aes(x=x), inherit.aes = FALSE, alpha=0.5) +
  geom_line(data = dat_theo, aes(x=x, y=y),inherit.aes = FALSE, col='skyblue')+
  ggtitle(paste0('MSE ours vs theoretical: ', round(mse,4)))


###############################################
# Dependency plots 
###############################################

######### x4 with flexible x2 given x1  #######
which(train$df_orig$numpy()[,1] < -1)
ln = 7
tn = 4
pn = 2
x1 = train$df_scaled$numpy()[ln,1]
x1
x1_org = train$df_orig$numpy()[ln,1]
r = train_data$parents[[tn]]
#hist(train$df_orig$numpy()[,2],100)
# Compute min and max of the second column of the old tensor
min_value = tf$reduce_min(r[,2L])
max_value = tf$reduce_max(r[,2L])
# Create a new tensor of the desired size and fill with required values
first_column = tf$fill(c(1000L, 1L), x1)
second_column = tf$reshape(tf$linspace(min_value, max_value, 1000L), c(-1L, 1L))
new_tensor = tf$concat(c(first_column, second_column), axis=1L)
res = sample_from_target(thetaNN_l[[tn]], new_tensor)
df = data.frame(x1=first_column$numpy(),x2=second_column$numpy(),x3=0, x4=res$numpy())
unscaled = unscale(train$df_orig, tf$constant(as.matrix(df), dtype=tf$float32))$numpy()
plot(unscaled[,2], unscaled[,tn], xlab='x2', ylab=paste0('x', tn), main = paste0('Given x1=',x1_org))
lines(unscaled[,2], lowess(unscaled[,4])$y, col='darkgreen', lwd=2)
rug(train$df_orig$numpy()[,2])
d = train$df_orig$numpy()[ln,]
abline(0.7*d[1]^2, -1, col='red', lwd=2)
points(d[2], 0.7*d[1]^2 - d[2], col='red', pch='+',cex=3)


######### x3 with flexible x1 given x2  #######
tn = 3 #The target number
pn = 1 #The flexible part
pf = 2 #The fixed part in 1,2,3,4
ln =1
x_fixed = train$df_scaled$numpy()[ln,c(pf)]
x_fixed
xpn_org = train$df_orig$numpy()[ln,pn]
r = train_data$parents[[tn]] #1,2
# Compute min and max of the second column of the old tensor
min_value = tf$reduce_min(r[,2L]) #TODO needs to be done by hand
max_value = tf$reduce_max(r[,2L])
# Create a new tensor of the desired size and fill with required values
first_column = tf$reshape(tf$linspace(min_value, max_value, 1000L), c(-1L, 1L)) #here x1 
second_column = tf$fill(c(1000L, 1L), x_fixed)
new_tensor = tf$concat(c(first_column, second_column), axis=1L)
res = sample_from_target(thetaNN_l[[tn]], new_tensor)
df = data.frame(x1=first_column$numpy(),x2=second_column$numpy(),x3=res$numpy(), x4=0)
unscaled = unscale(train$df_orig, tf$constant(as.matrix(df), dtype=tf$float32))$numpy()

plot(unscaled[,pn], unscaled[,tn], xlab=paste0('x', pn), ylab=paste0('x', tn), main = paste0('Given x_fixed=',x_fixed))
lines(unscaled[,pn], lowess(unscaled[,tn])$y, col='darkgreen', lwd=4)
rug(train$df_orig$numpy()[,2])
d = train$df_orig$numpy()[ln,]
#x3=x1+c1*x2^3
abline(0.3*d[pf]^2, 1, col='red', lwd=2)
points(d[pn], 0.3*d[pf]^2 + d[pn], col='red', pch='+',cex=3)


######### x4 with flexible x1 given x2  #######
tn = 4 #The target number
pn = 1 #The flexible part
pf = 2 #The fixed part in 1,2,3,4
ln = 1
x_fixed = train$df_scaled$numpy()[ln,c(pf)]
x_fixed
xpn_org = train$df_orig$numpy()[ln,pn]
r = train_data$parents[[tn]]
# Compute min and max of the second column of the old tensor
min_value = tf$reduce_min(r[,1L]) #TODO needs to be done by hand
max_value = tf$reduce_max(r[,1L])
# Create a new tensor of the desired size and fill with required values
first_column = tf$reshape(tf$linspace(min_value, max_value, 1000L), c(-1L, 1L)) #here x1 
second_column = tf$fill(c(1000L, 1L), x_fixed)
new_tensor = tf$concat(c(first_column, second_column), axis=1L)
res = sample_from_target(thetaNN_l[[tn]], new_tensor)
df = data.frame(x1=first_column$numpy(),x2=second_column$numpy(),x3=0, x4=res$numpy())
unscaled = unscale(train$df_orig, tf$constant(as.matrix(df), dtype=tf$float32))$numpy()

plot(unscaled[,pn], unscaled[,tn], xlab=paste0('x', pn), ylab=paste0('x', tn), main = paste0('Given x_fixed=',x_fixed))
lines(unscaled[,pn], lowess(unscaled[,tn])$y, col='darkgreen', lwd=4)
rug(train$df_orig$numpy()[,2])
d = train$df_orig$numpy()[ln,]
#c1
lines(unscaled[,pn], coeffs[2]*unscaled[,pn]^2 -d[pf], col='red', lwd=2)
points(d[pn], coeffs[2]*d[pn]^2 - d[pf], col='red', pch='+',cex=3)












