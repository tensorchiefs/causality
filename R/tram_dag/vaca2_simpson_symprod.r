source('R/tram_dag/utils.R')
library(R.utils)
DEBUG = FALSE
DEBUG_NO_EXTRA = FALSE
USE_EXTERNAL_DATA = FALSE 
M = 30
SUFFIX = 'run_simpson_symprod_10k_2500_M30'
EPOCHS = 10000

DROPBOX = 'C:/Users/sick/dl Dropbox/beate sick/IDP_Projekte/DL_Projekte/shared_Oliver_Beate/Causality_2022/tram_DAG/'
DROPBOX = '~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/Causality_2022/tram_DAG/'
if (rstudioapi::isAvailable()) {
  context <- rstudioapi::getSourceEditorContext()
  this_file <- context$path
  print(this_file)
} else{
  this_file = "~/Documents/GitHub/causality/R/tram_dag/vaca2_triangle_nl.r"
}

#######################
# Latent Distribution
latent_dist = tfd_logistic(loc=0, scale=1)
#latent_dist = tfd_normal(loc=0, scale=1)
#latent_dist = tfd_truncated_normal(loc=0., scale=1.,low=-4,high = 4)


len_theta = M + 1
bp = make_bernp(len_theta)

######################################
############# DGP ###############
######################################
dgp <- function(n_obs, coeffs = NULL, doX1=NA, dat_train=NULL, seed=NA, file=NULL) {
  if (is.na(seed) == FALSE){
    set.seed(seed)
  }
  
  #Use external data 
  if (is.null(file) == FALSE){
    stop("Not implemented yet")
    #data <- read.csv(file, header = FALSE)
    #X_1 <- data[,1]
    #X_2 <- data[,2]
    #X_3 <- data[,3]
    #X_4 <- data[,4]
    #n_obs=length(X_4)
  } else{
    X_1 = rnorm(n_obs)
    if (is.na(doX1) == FALSE){
      X_1 = X_1 * 0 + doX1
    }
    X_2 = 2*tanh(2*X_1) + 1/sqrt(10) * rnorm(n_obs)
    X_3 = 0.5*X_1 * X_2 + 1/sqrt(2) * rnorm(n_obs)
    X_4 = tanh(1.5*X_1) + sqrt(3/10) * rnorm(n_obs)
  }
  
  dat.s =  data.frame(x1 = X_1, x2 = X_2, x3 = X_3, x4=X_4)
  dat.tf = tf$constant(as.matrix(dat.s), dtype = 'float32')
  A <- matrix(c(0, 1, 1, 1, 0,0,1,0, 0,0,0,0, 0,0,0,0), nrow = 4, ncol = 4, byrow = TRUE)
  
  if (is.null(dat_train)){
    scaled = scale_df(dat.tf)
  } else{
    scaled = scale_validation(dat_train, dat.tf)
  }
  return(list(df_orig=dat.tf, df_scaled = scaled, coef=coeffs, A=A, name='vaca2_simpson_symprod'))
} 

if (USE_EXTERNAL_DATA){
  stop("Not implemted (no external data)")
} else{
  train = dgp(2500, seed=42)
}

pairs(train$df_orig$numpy())
par(mfrow=c(2,2))
hist(train$df_orig$numpy()[,1],100)
hist(train$df_orig$numpy()[,2],100)
hist(train$df_orig$numpy()[,3],100)
hist(train$df_orig$numpy()[,4],100)
par(mfrow=c(1,1))
pairs(train$df_scaled$numpy())
train$A
library(igraph)
graph <- graph_from_adjacency_matrix(train$A, mode = "directed", diag = FALSE)
plot(graph, vertex.color = "lightblue", vertex.size = 30, edge.arrow.size = 0.5)

train_data = split_data(train$A, train$df_scaled)
thetaNN_l = make_thetaNN(train$A, train_data$parents)

if(USE_EXTERNAL_DATA){
  val = train
} else{
  val = dgp(5000, dat_train = train$df_orig)
}

val_data = split_data(val$A, val$df_scaled)


###### Training Step #####
optimizer = tf$keras$optimizers$Adam(learning_rate=0.001)
l = do_training(train$name, thetaNN_l = thetaNN_l, train_data = train_data, val_data = val_data,
                SUFFIX, epochs = EPOCHS,  optimizer=optimizer, dynamic_lr = TRUE)

### Save Script
dirname = paste0(DROPBOX, "exp/", train$name, "/", SUFFIX, "/")
file.copy(this_file, dirname)

loss = l[[1]]
loss_val = l[[2]]
plot(loss, type='l', ylim=c(-4.5,-3.0))
points(loss_val, col='green')

#Loading data from epoch e
load_weights(epoch = EPOCHS, l)

##########. Checking obs fit the marginals 
plot_obs_fit(train_data$parents, train_data$target, thetaNN_l, name='Training')
plot_obs_fit(val_data$parents, val_data$target, thetaNN_l, name='Validation')

############################### Do X via Flow ########################
########################
# Do Interventions on x1 #####
dox_origs = seq(-2, 2, by = 0.5)
num_samples = 1142L
inter_mean_dgp_x2 = inter_mean_dgp_x3 = 
  inter_mean_dgp_x4 = inter_mean_ours_x2 = 
  inter_mean_ours_x3 = inter_mean_ours_x4 = NA*dox_origs

inter_dgp_x2 = inter_dgp_x3 = inter_dgp_x4 =
  inter_ours_x2 = inter_ours_x3 = inter_ours_x4 = 
  matrix(NA, nrow=length(dox_origs), ncol=num_samples)

dox_origs_simu = seq(-2, 2, by = 0.5)
for (i in 1:length(dox_origs)){
  dox_orig = dox_origs[i]
  #dox_orig = -2 # we expect E(X3|X1=dox_orig)=dox_orig
  dox=scale_value(train$df_orig, col=1L, dox_orig)
  #dat_do_x_s = dox1(dox, thetaNN_l, num_samples = num_samples)
  dat_do_x_s = do(thetaNN_l, A = train$A, doX = c(dox, NA, NA, NA), num_samples)
  
  #
  df = unscale(train$df_orig, dat_do_x_s)
  inter_ours_x2[i,] = df$numpy()[,2]
  inter_ours_x3[i,] = df$numpy()[,3]
  inter_ours_x4[i,] = df$numpy()[,4]
  
  inter_mean_ours_x2[i] = mean(df[,2]$numpy())
  inter_mean_ours_x3[i] = mean(df[,3]$numpy())
  inter_mean_ours_x4[i] = mean(df[,4]$numpy())
  
  #res_med_x4[i] = median(df[,4]$numpy())
  
  d = dgp(num_samples,doX1=dox_orig)
  inter_mean_dgp_x2[i] = mean(d$df_orig[,2]$numpy())
  inter_mean_dgp_x3[i] = mean(d$df_orig[,3]$numpy())
  inter_mean_dgp_x4[i] = mean(d$df_orig[,4]$numpy())
  
  inter_dgp_x2[i,] = d$df_orig[,2]$numpy()
  inter_dgp_x3[i,] = d$df_orig[,3]$numpy()
  inter_dgp_x4[i,] = d$df_orig[,4]$numpy()
}

## Interventional Distributions ####
df_do = data.frame(dox=numeric(0),x2=numeric(0),x3=numeric(0), x4=numeric(0),type=character(0))
for (step in c(1,3,5,6,9)){
    df_do = rbind(df_do, data.frame(
      dox = dox_origs[step],
      x2 = inter_dgp_x2[step,],
      x3 = inter_dgp_x3[step,],
      x4 = inter_dgp_x4[step,],
      type = 'simu'
    ))
    df_do = rbind(df_do, data.frame(
      dox = dox_origs[step],
      x2 = inter_ours_x2[step,],
      x3 = inter_ours_x3[step,],
      x4 = inter_ours_x4[step,],
      type = 'ours'
    )
  )
}

### Plotting the Dists ####
df_do$facet_label <- paste("dox1 =", df_do$dox)
ggplot(df_do) + 
  geom_density(aes(x=x2, col=type, linetype=type)) + 
  ylab("p(x2|do(x1)") +
  facet_grid(~facet_label)
ggsave(make_fn("dox1_dist_x2.pdf"))

ggplot(df_do) + 
  geom_density(aes(x=x3, col=type, linetype=type)) + 
  ylab("p(x3|do(x1)") +
  facet_grid(~facet_label)
ggsave(make_fn("dox1_dist_x3.pdf"))

ggplot(df_do) + 
  geom_density(aes(x=x4, col=type, linetype=type)) + 
  ylab("p(x4|do(x1)") +
  facet_grid(~as.factor(facet_label))
ggsave(make_fn("dox1_dist_x4.pdf"))


### Plotting the mean effects ####
x1dat = data.frame(x=train$df_orig$numpy()[,1])
library(ggplot2)

### X2
df_do_mean = data.frame(
  x = dox_origs,
  Ours = inter_mean_ours_x2,
  #theoretical = dox_origs,
  Simulation_DGP = inter_mean_dgp_x2
) 
d = df_do_mean %>% 
  pivot_longer(cols = 2:3)

ggplot() + 
  geom_point(data = subset(d, name == "Ours"), aes(x=x, y=value, col=name)) +
  geom_line(data = subset(d, name == "Simulation_DGP"), aes(x=x, y=value, col=name, type=name))+
  #geom_abline(intercept = 0, slope = 1, col='skyblue') +
  xlab("do(X1)") +
  ylab("E(X2|do(X1)") +
  geom_rug(data=x1dat, aes(x=x), inherit.aes = FALSE, alpha=0.5)

ggsave(make_fn("dox1_mean_x2.pdf"))

### X3
df_do_mean = data.frame(
  x = dox_origs,
  Ours = inter_mean_ours_x3,
  #theoretical = dox_origs,
  Simulation_DGP = inter_mean_dgp_x3
) 
d = df_do_mean %>% 
  pivot_longer(cols = 2:3)

ggplot() + 
  geom_point(data = subset(d, name == "Ours"), aes(x=x, y=value, col=name)) +
  geom_line(data = subset(d, name == "Simulation_DGP"), aes(x=x, y=value, col=name, type=name))+
  #geom_abline(intercept = 0, slope = 1, col='skyblue') +
  xlab("do(X1)") +
  ylab("E(X3|do(X1)") +
  geom_rug(data=x1dat, aes(x=x), inherit.aes = FALSE, alpha=0.5) 

ggsave(make_fn("dox1_mean_x3.pdf"))

### X4
df_do_mean = data.frame(
  x = dox_origs,
  Ours = inter_mean_ours_x4,
  #theoretical = dox_origs,
  Simulation_DGP = inter_mean_dgp_x4
) 
d = df_do_mean %>% 
  pivot_longer(cols = 2:3)

ggplot() + 
  geom_point(data = subset(d, name == "Ours"), aes(x=x, y=value, col=name)) +
  geom_line(data = subset(d, name == "Simulation_DGP"), aes(x=x, y=value, col=name, type=name))+
  #geom_abline(intercept = 0, slope = 1, col='skyblue') +
  xlab("do(X1)") +
  ylab("E(X4|do(X1)") +
  geom_rug(data=x1dat, aes(x=x), inherit.aes = FALSE, alpha=0.5) 

ggsave(make_fn("dox1_mean_x4.pdf"))


#######################
#### Counterfactua ####
# CF of X4 given X1=alpha ###

#Creating a typical value 
if (FALSE){
  X1 = -0.5
  x1 = scale_value(dat_train_orig = train$df_orig, col = 1, value = X1)$numpy()
  x2 = get_x(thetaNN_l[[2]], x1, 0.)
  x3 = get_x(thetaNN_l[[3]], c(x1,x2), 0.)
  x4 = get_x(thetaNN_l[[4]], c(x1), 0.)
  df = data.frame(x1=0,x2=as.numeric(x2), x3=as.numeric(x3),x4=as.numeric(x4))
  unscaled = unscale(train$df_orig, tf$constant(as.matrix(df), dtype=tf$float32))$numpy()
  X2 = unscaled[2]
  X3 = unscaled[3]
  X4 = unscaled[4]
  
  dat = val$df_orig$numpy()
  point = c(X1,X2,X3,X4)
  distances <- apply(dat, 1, function(row) sqrt(sum((row - point)^2)))
  # Find the index of the closest row
  closest_row_index <- which.min(distances)
  # Retrieve the closest row
  closest_row <- dat[closest_row_index,]
  closest_row #-0.4758347 -1.4321059  0.3376575 -0.5187235
} else{
  X1 = -0.4758347
  X2 = -1.4321059 
  X3 = 0.3376575
  X4 = -0.5187235
  X_obs = c(X1, X2, X3, X4)
}

###### CF Theoretical (assume we know the complete SCM) #####


cf_do_x1_dgp = function(alpha){
  #Abduction these are the Us that correspond the observed values
  U1 = X1 
  U2 = (X2 - 2*tanh(2*X1))*sqrt(10)
  U3 = (X3 - 0.5*X1*X2)*sqrt(2)
  U4 = (X4 - tanh(1.5*X1)) * sqrt(10/3)
  #X1 --> alpha
  X_1 = alpha
  X_2 = 2*tanh(2*X_1) + 1/sqrt(10) * U2
  X_3 = 0.5*X_1 * X_2 + 1/sqrt(2) * U3
  X_4 = tanh(1.5*X_1) + sqrt(3/10) * U4
  return(data.frame(X1=X_1,X2=X_2,X3=X_3, X4=X_4))
}

#Constistency
cf_dgp_cons = cf_do_x1_dgp(X1)
abs(cf_dgp_cons$X2-X2) #~1e-16 Consistency
abs(cf_dgp_cons$X3-X3) #~1e-16 Consistency
abs(cf_dgp_cons$X4-X4) #~1e-16 Consistency
cf_do_x1_dgp(0)
cf_do_x1_dgp(1)

##### Our Approach ####
xobs = scale_validation(train$df_orig, X_obs)$numpy()

## Checking for consistency
computeCF(thetaNN_l, A=train$A, xobs = xobs, cfdoX = c(NA, NA,NA,NA)) - xobs
computeCF(thetaNN_l, A=train$A, xobs = xobs, cfdoX = c(xobs[1], NA,NA,NA)) - xobs
computeCF(thetaNN_l, A=train$A, xobs = xobs, cfdoX = c(NA, xobs[2],NA,NA)) - xobs
computeCF(thetaNN_l, A=train$A, xobs = xobs, cfdoX = c(NA, NA,xobs[3],NA)) - xobs
computeCF(thetaNN_l, A=train$A, xobs = xobs, cfdoX = c(NA, NA,NA,xobs[4])) - xobs


## Creating Results for do(x1)
df = data.frame()
for (a_org in c(seq(-3,3,0.2),X1)){
  #a_org = 2
  dgp = cf_do_x1_dgp(a_org)
  df = bind_rows(df, data.frame(x1=a_org, X2=dgp[2], X3=dgp[3],X4=dgp[4], type='DGP'))
  
  a = scale_value(train$df_orig, 1L, a_org)$numpy()
  printf("a_org %f a %f \n", a_org, a)
  #cf_our = cf_do_x1_ours(a_org)
  cf_our = computeCF(thetaNN_l = thetaNN_l, A = train$A, cfdoX = c(a,NA,NA,NA), xobs = xobs)
  cf_our = unscale(train$df_orig, matrix(cf_our, nrow=1))$numpy()
  df = bind_rows(df, data.frame(x1=a_org, X2=cf_our[2], X3=cf_our[3], X4=dgp[4], type='OURS'))
}

library(ggpubr)
ggplot(df) +
  geom_point(data = subset(df, type == "OURS"), aes(x = x1, y = X2, color=type)) +
  geom_line(data = subset(df, type == "DGP"), aes(x = x1, y = X2, color=type)) + 
  xlab('would x1 be alpha')  + 
  geom_rug(data=x1dat, aes(x=x), inherit.aes = FALSE, alpha=0.5) +
  theme_pubr() +  # Positioning the legend to the lower right corner
  labs(color = "") 

ggsave(make_fn("CFx1_x2.pdf"))


ggplot(df) +
  geom_point(data = subset(df, type == "OURS"), aes(x = x1, y = X3, color=type)) +
  geom_line(data = subset(df, type == "DGP"), aes(x = x1, y = X3, color=type)) + 
  xlab('would x1 be alpha')  + 
  geom_rug(data=x1dat, aes(x=x), inherit.aes = FALSE, alpha=0.5) +
  theme_pubr() +  # Positioning the legend to the lower right corner
  labs(color = "") 

ggsave(make_fn("CFx1_x3.pdf"))

#

ggplot(df) +
  geom_point(data = subset(df, type == "OURS"), aes(x = x1, y = X4, color=type)) +
  geom_line(data = subset(df, type == "DGP"), aes(x = x1, y = X4, color=type)) + 
  xlab('would x1 be alpha')  + 
  geom_rug(data=x1dat, aes(x=x), inherit.aes = FALSE, alpha=0.5) +
  theme_pubr() +  # Positioning the legend to the lower right corner
  labs(color = "") 

ggsave(make_fn("CFx1_x4.pdf"))




########################
# Old Stuff for checking
if (FALSE){
  dox1 = function(doX, thetaNN_l, num_samples){
    doX_tensor = doX * tf$ones(shape=c(num_samples,1L),dtype=tf$float32) 
    
    x2_samples = sample_from_target(thetaNN_l[[2]], doX_tensor)
    x4_samples = sample_from_target(thetaNN_l[[4]], doX_tensor)
    
    parents_x3 = tf$concat(list(doX_tensor, x2_samples), axis=1L)
    x3_samples = sample_from_target(thetaNN_l[[3]], parents_x3)
    
    
    return(matrix(c(doX_tensor$numpy(),x2_samples$numpy(), 
                    x3_samples$numpy(), x4_samples$numpy()), ncol=4))
  }
  
  df = dox1(0.5, thetaNN_l, num_samples=1e4L)
  str(df)
  summary(df)
  
  df2t = do(thetaNN_l, train$A, doX=c(0.5, NA, NA, NA), num_samples=1e4)
  
  df2 = as.matrix(df2t$numpy())
  qqplot(df2[,2], df[,2]);abline(0,1)
  qqplot(df2[,3], df[,3]);abline(0,1)
  qqplot(df2[,4], df[,4]);abline(0,1)
}

if (FALSE){
  ###### CF From our model #####
  # Scaling the observed data so that it can be used with the networks
  cf_do_x1_ours = function(alpha){
    
    # Abduction (no need for alpha)
    ## Scaling of the observed variables
    x1 = scale_value(dat_train_orig = train$df_orig, col = 1, value = X1)$numpy()
    x2 = scale_value(dat_train_orig = train$df_orig, col = 2, value = X2)$numpy()
    x3 = scale_value(dat_train_orig = train$df_orig, col = 3, value = X3)$numpy()
    x4 = scale_value(dat_train_orig = train$df_orig, col = 4, value = X4)$numpy()
    
    # Abduction Getting the relevant latent variable
    #Z1 Not needed later
    z1 = get_z(thetaNN_l[[1]], parents = 1L, x=x2)
    z2 =  get_z(thetaNN_l[[2]], parents = x1, x=x2)
    z3 =  get_z(thetaNN_l[[3]], parents = c(x1, x2), x=x3)
    z4 =  get_z(thetaNN_l[[4]], parents = x1, x=x4)
    
    ## Action and prediction
    #a = scale_value(dat_train_orig = train$df_orig, col = 1, value = alpha)$numpy()
    a = alpha 
    x2_CF = get_x(net=thetaNN_l[[2]], parents = a, z2) #X1-->a but the same latent variable
    x3_CF = get_x(net=thetaNN_l[[3]], parents = c(a, x2_CF), z3) 
    x4_CF = get_x(net=thetaNN_l[[4]], parents = a, z4) 
    
    df = data.frame(x1=x1, x2=as.numeric(x2_CF),x3=as.numeric(x3_CF),x4=as.numeric(x4_CF))
    #unscaled = unscale(train$df_orig, tf$constant(as.matrix(df), dtype=tf$float32))$numpy()
    return(df)
  }
  
  
  #Consistency
  cf_do_x1_ours(0.5)
  
  #
  xobs = scale_validation(train$df_orig, X_obs)
  computeCF(thetaNN_l, A=train$A, xobs = xobs$numpy(), cfdoX = c(0.5, NA,NA,NA))
  
}





