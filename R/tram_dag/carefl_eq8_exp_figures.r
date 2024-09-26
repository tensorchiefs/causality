library(R.utils)
source('R/tram_dag/utils.R')
source('configs/configs_defaults.r')
source('configs/config_user.r')
source('configs/config_carefl_eq8.R')

#config_user("beate_server")
config_user("oliver")

short_name = "carefl_eq8"
( SUFFIX = get_suffix(short_name ) )

SUFFIX = "runNormal_M30_carefldata_scaled_e10000"  # set if pointing to a trained model

if (rstudioapi::isAvailable()) {
  context <- rstudioapi::getSourceEditorContext()
  this_file <- context$path
  print(this_file)
} else{
  this_file = "~/Documents/GitHub/causality/R/tram_dag/carefl_eq8.r"
}


len_theta = M + 1
bp = make_bernp(len_theta)


#Data from CAREFL Fig 5
if (USE_EXTERNAL_DATA){

  file = EXTERNAL_DATA_FILE
  train = dgp(file = file, coeffs = coeffs, seed=42)
  #We compare against the results written in counterfactual_trials.py
  colMeans(train$df_orig$numpy()) #CAREFL [-0.01598893  0.00515872  0.00869695  0.26056851]
  apply(train$df_orig$numpy(), 2, sd) #CAREFL [1.03270839 0.98032533 1.         1.        ]
  X_obs <- read.csv('data/CAREFL_CF/xObs.csv', header = FALSE)[1,]
  X_obs = as.numeric(X_obs)
  X_obs #[[ 2.          1.5         0.84645028 -0.26158623]]
} else{
  train = dgp(nTrain, coeffs = coeffs, seed=42)
}

pairs(train$df_orig$numpy())
pairs(train$df_scaled$numpy())
train$coef
train$A


check_avail = require("igraph")
if( check_avail )
{
  library(igraph)
  graph <- graph_from_adjacency_matrix(train$A, mode = "directed", diag = FALSE)
  plot(graph, vertex.color = "lightblue", vertex.size = 30, edge.arrow.size = 0.5)
}
  

train_data = split_data(train$A, train$df_scaled)
thetaNN_l = make_thetaNN(train$A, train_data$parents)

if(USE_EXTERNAL_DATA){
  val = train
} else{
  val = dgp(5000, coeffs = coeffs, dat_train = train$df_orig)
}

val_data = split_data(val$A, val$df_scaled)


###### Training Step #####

l = do_training(train$name, thetaNN_l = thetaNN_l, train_data = train_data, val_data = val_data,
                SUFFIX, epochs = EPOCHS,  optimizer=optimizer)

### Save Script
dirname = paste0(DROPBOX, "exp/", train$name, "/", SUFFIX, "/")
file.copy(this_file, dirname)
file.copy(file.path(getwd(),"configs/configs_defaults.r"), dirname)
file.copy(file.path(getwd(),"configs/config_user.r"), dirname)
file.copy(file.path(getwd(),"configs/config_carefl_eq8.R"), dirname)


loss = l[[1]]
loss_val = l[[2]]
plot(loss, type='l', ylim=c(-8.0,-3))
points(loss_val, col='green')

#Loading data from epoch e
load_weights(epoch = EPOCHS, l)


##########. Checking obs fit the marginals 
# plot_obs_fit(train_data$parents, train_data$target, thetaNN_l, name='Training')
# plot_obs_fit(val_data$parents, val_data$target, thetaNN_l, name='Validation')


############################### Do X via Models ########################

# load interventional distribution as estimated by Carefl-Model in python
library(reticulate)
load_pickle <- function(filepath) {
  
  
  reticulate::py_run_string(paste0("
import pickle
with open('", filepath, "', 'rb') as file:
    data = pickle.load(file)
  "))
  
  return(py$data)
}

data <- load_pickle('data/int_2500r_cl_mlp_5_10.p')
summary(data)

##############################
# Do Interventions on x1 #####
dox_origs = seq(-3, 3, by = 0.5)
res_med_x4  = res_scm_x4 = res_scm_x3 = res_x3 = res_x4 = dox_origs
for (i in 1:length(dox_origs)){
  
  dox_orig = dox_origs[i]
  dox=scale_value(train$df_orig, col=1L, dox_orig)
  num_samples = 5142L
  #dat_do_x_s = dox1_eq8(dox, thetaNN_l, num_samples = num_samples)
  dat_do_x_s = do(thetaNN_l, train$A, doX=c(dox, NA, NA, NA), num_samples=num_samples)
  df = unscale(train$df_orig, dat_do_x_s)
  res_x3[i] = mean(df[,3]$numpy())
  res_x4[i] = mean(df[,4]$numpy())
 
  d = dgp(5000L, coeffs = coeffs, doX1=dox_orig)
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
##### Counterfactual  #########################
###############################################
### CF of X4 given X1=alpha ####

# Attention, we don't have the same C2 as in the paper
# train$df_orig[2,]
# #Paper
# X1 = 2
# X2 = 1.5
# X3 = 0.81
# X4 = -0.28

#Creating a typical value 
if (FALSE){
  X1 = 0.5
  X2 = -0.5
  x1 = scale_value(dat_train_orig = train$df_orig, col = 1, value = X1)$numpy()
  x2 = scale_value(dat_train_orig = train$df_orig, col = 1, value = X2)$numpy()
  x3 = get_x(thetaNN_l[[3]], c(x1,x2), 0.)
  x4 = get_x(thetaNN_l[[4]], c(x1,x2), 0.)
  df = data.frame(x1=0,x2=0,x3=as.numeric(x3), x4=as.numeric(x4))
  unscaled = unscale(train$df_orig, tf$constant(as.matrix(df), dtype=tf$float32))$numpy()
  X3 = unscaled[3]
  X4 = unscaled[4]
} else if (USE_EXTERNAL_DATA == FALSE){
  X1 = 0.5
  X2 = -0.5
  X3 = -0.243927
  X4 = 1.137823
  X_obs = c(X1, X2, X3, X4)
}
if (USE_EXTERNAL_DATA){
  X1 = X_obs[1]
  X2 = X_obs[2]
  X3 = X_obs[3]
  X4 = X_obs[4]
}

cf_do_x1_dgp = function(alpha){
    ###### Theoretical (assume we know the complete SCM)
    # X_3 <- X_1 + coeffs[1] * (X_2^3)  + U3
    # X_4 <- -X_2 + coeffs[2] * (X_1^2) + U4
    
    U1 = X1
    U2 = X2
    U3 = X3 - X1 - coeffs[1] * X2^3
    U4 = X4 - coeffs[2]*X1^2 + X2
    #X1 --> alpha
    X_1 = alpha
    X_2 = U2
    X_3 = X_1 + coeffs[1] * (X_2^3)  + U3
    X_4 = -X_2 + coeffs[2] * (X_1^2) + U4
   return(data.frame(X1=X_1,X2=X_2,X3=X_3,X4=X_4))
}

abs(cf_do_x1_dgp(X_obs[1])-X_obs) #~1e-16 Consistency

###### From our model
xobs = scale_validation(train$df_orig, X_obs)$numpy()
computeCF(thetaNN_l, A=train$A, xobs = xobs, cfdoX = c(NA, NA,NA,NA)) - xobs
computeCF(thetaNN_l, A=train$A, xobs = xobs, cfdoX = c(xobs[1], NA,NA,NA)) - xobs
computeCF(thetaNN_l, A=train$A, xobs = xobs, cfdoX = c(NA, xobs[2],NA,NA)) - xobs
computeCF(thetaNN_l, A=train$A, xobs = xobs, cfdoX = c(NA, NA,xobs[3],NA)) - xobs
computeCF(thetaNN_l, A=train$A, xobs = xobs, cfdoX = c(NA, NA,NA,xobs[4])) - xobs

## Creating Results for computeCF(x1)
df = data.frame()
for (a_org in c(seq(-3,3,0.05),X1)){
  dgp = cf_do_x1_dgp(a_org)
  df = bind_rows(df, data.frame(x1=a_org, X2=dgp[2], X3=dgp[3], X4=dgp[4], type='DGP'))
}

for (a_org in c(seq(-3,3,0.2),X1)){
  a = scale_value(train$df_orig, 1L, a_org)$numpy()
  printf("a_org %f a %f \n", a_org, a)
  #cf_our = cf_do_x1_ours(a_org)
  cf_our = computeCF(thetaNN_l = thetaNN_l, A = train$A, cfdoX = c(a,NA,NA,NA), xobs = xobs)
  cf_our = unscale(train$df_orig, matrix(cf_our, nrow=1))$numpy()
  df = bind_rows(df, data.frame(x1=a_org, X2=cf_our[2], X3=cf_our[3], X4=cf_our[4], type='OURS'))
}

x1dat = data.frame(x=train$df_orig$numpy()[,1])
ggplot(df) +
  geom_point(data = subset(df, type == "OURS"), aes(x = x1, y = X4, color=type)) +
  geom_line(data = subset(df, type == "DGP"), aes(x = x1, y = X4, color=type)) + 
  geom_rug(data=x1dat, aes(x=x), inherit.aes = FALSE, alpha=0.5) +
  xlab('would x1 be alpha')  

ggplot(df) +
  geom_point(data = subset(df, type == "OURS"), aes(x = x1, y = X3, color=type)) +
  geom_line(data = subset(df, type == "DGP"), aes(x = x1, y = X3, color=type)) + 
  geom_rug(data=data.frame(x=train$df_orig$numpy()[,1]), aes(x=x), inherit.aes = FALSE, alpha=0.5) +
  xlab('would x1 be alpha')  



###############################################
# Counterfactual  
###############################################
#X3 --> alpha
cf_do_x2_dgp = function(alpha){
  ###### Theoretical (assume we know the complete SCM)
  # X_3 <- X_1 + coeffs[1] * (X_2^3)  + U3
  # X_4 <- -X_2 + coeffs[2] * (X_1^2) + U4
  U1 = X1
  U2 = X2
  U3 = X3 - X1 - coeffs[1] * X2^3
  U4 = X4 - coeffs[2]*X1^2 + X2
  #X1 --> alpha
  X_1 = U1
  X_2 = alpha
  X_3 = X_1 + coeffs[1] * (X_2^3)  + U3
  X_4 = -X_2 + coeffs[2] * (X_1^2) + U4
  return(data.frame(X1=X_1,X2=X_2,X3=X_3,X4=X_4))
}

abs(cf_do_x2_dgp(X2)-X_obs) #~1e-16 Consistency

## Creating Results for do(x2)
df = data.frame()
for (a_org in c(seq(-3.5,3.5,0.05),X2)){
  dgp = cf_do_x2_dgp(a_org)
  df = bind_rows(df, data.frame(x1=dgp[1], X2=a_org, X3=dgp[3], X4=dgp[4], type='DGP'))
}

for (a_org in c(seq(-3.5,3.5,0.5),X2)){
  a = scale_value(train$df_orig, 2L, a_org)$numpy()
  printf("a_org %f a %f \n", a_org, a)
  cf_our = computeCF(thetaNN_l = thetaNN_l, A = train$A, cfdoX = c(NA,a,NA,NA), xobs = xobs)
  cf_our = unscale(train$df_orig, matrix(cf_our, nrow=1))$numpy()
  df = bind_rows(df, data.frame(X1=a_org, X2=cf_our[2], X3=cf_our[3], X4=cf_our[4], type='OURS'))
}

ggplot(df) +
  geom_point(data = subset(df, type == "OURS"), aes(x = X2, y = X4, color=type)) +
  geom_line(data = subset(df, type == "DGP"), aes(x = X2, y = X4, color=type)) + 
  geom_rug(data=data.frame(x=train$df_orig$numpy()[,2]), aes(x=x), inherit.aes = FALSE, alpha=0.5) +
  xlab('would x2 be alpha')  

ggplot(df) +
  geom_point(data = subset(df, type == "OURS"), aes(x = X2, y = X3, color=type)) +
  geom_line(data = subset(df, type == "DGP"), aes(x = X2, y = X3, color=type)) + 
  geom_rug(data=data.frame(x=train$df_orig$numpy()[,2]), aes(x=x), inherit.aes = FALSE, alpha=0.5) +
  xlab('would x2 be alpha')  


#################
# Old do stuff ##
#################

if (FALSE){
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

df = dox1_eq8(0.5, thetaNN_l, num_samples=1e4L)
str(df)
summary(df)

df2t = do(thetaNN_l, train$A, doX=c(0.5, NA, NA, NA), num_samples=1e4)
df2 = as.matrix(df2t$numpy())
qqplot(df2[,2], df[,2]);abline(0,1)
qqplot(df2[,3], df[,3]);abline(0,1)
qqplot(df2[,4], df[,4]);abline(0,1)

}

###############################################
# Dependency plots 
###############################################
if (FALSE){
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
}











