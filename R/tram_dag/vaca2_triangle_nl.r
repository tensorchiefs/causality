source('R/tram_dag/utils.R')
library(R.utils)
DEBUG = FALSE
DEBUG_NO_EXTRA = FALSE
USE_EXTERNAL_DATA = FALSE 
SUFFIX = 'run_triangle_NL_dynamik_long10k_M30_run01'
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

M = 30
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
    X_1 = 1 + rnorm(n_obs)
    if (is.na(doX1) == FALSE){
      X_1 = X_1 * 0 + doX1
    }
    X_2 = 2*X_1^2 + rnorm(n_obs)
    X_3 = 20./(1 + exp(-X_2^2 + X_1)) + rnorm(n_obs)
  }
  
  dat.s =  data.frame(x1 = X_1, x2 = X_2, x3 = X_3)
  dat.tf = tf$constant(as.matrix(dat.s), dtype = 'float32')
  A <- matrix(c(0, 1, 1, 0,0,1,0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
  
  if (is.null(dat_train)){
    scaled = scale_df(dat.tf)
  } else{
    scaled = scale_validation(dat_train, dat.tf)
  }
  return(list(df_orig=dat.tf, df_scaled = scaled, coef=coeffs, A=A, name='vaca2_triangle_nl'))
} 

#Data from CAREFL Fig 5
if (USE_EXTERNAL_DATA){
  stop("Not implemted (no external data)")
} else{
  train = dgp(2500, seed=42)
}

pairs(train$df_orig$numpy())
par(mfrow=c(1,3))
hist(train$df_orig$numpy()[,1],100)
hist(train$df_orig$numpy()[,3],100)
hist(train$df_orig$numpy()[,2],100)
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
                SUFFIX, epochs = EPOCHS,  optimizer=optimizer)

### Save Script
dirname = paste0(DROPBOX, "exp/", train$name, "/", SUFFIX, "/")
file.copy(this_file, dirname)

loss = l[[1]]
loss_val = l[[2]]
plot(loss, type='l', ylim=c(-4.5,-4.3))
points(loss_val, col='green')

#Loading data from epoch e
load_weights(epoch = EPOCHS, l)

##########. Checking obs fit the marginals 
plot_obs_fit(train_data$parents, train_data$target, thetaNN_l, name='Training')
plot_obs_fit(val_data$parents, val_data$target, thetaNN_l, name='Validation')

############################### Do X via Flow ########################
#Samples from Z give X=doX


########################
# Do Interventions on x1 #####
dox_origs = seq(-3, 4, by = 0.5)
#dox_origs = seq(0, 1, by = 1)
num_samples = 1142L
inter_mean_dgp_x2 = inter_mean_dgp_x3 = inter_mean_ours_x2 = inter_mean_ours_x3 = NA*dox_origs

inter_dgp_x2 = inter_dgp_x3 = inter_ours_x2 = inter_ours_x3 = matrix(NA, nrow=length(dox_origs), ncol=num_samples)
for (i in 1:length(dox_origs)){
  dox_orig = dox_origs[i]
  #dox_orig = -2 # we expect E(X3|X1=dox_orig)=dox_orig
  dox=scale_value(train$df_orig, col=1L, dox_orig)
  
  #dat_do_x_s = dox1(dox, thetaNN_l, num_samples = num_samples)
  dat_do_x_s = do(thetaNN_l, train$A, doX = c(dox,NA,NA), num_samples = num_samples)
  
  #
  df = unscale(train$df_orig, dat_do_x_s)
  inter_ours_x2[i,] = df$numpy()[,2]
  inter_ours_x3[i,] = df$numpy()[,3]
  inter_mean_ours_x2[i] = mean(df[,2]$numpy())
  inter_mean_ours_x3[i] = mean(df[,3]$numpy())
  #res_med_x4[i] = median(df[,4]$numpy())
  
  
  d = dgp(num_samples,doX1=dox_orig)
  inter_mean_dgp_x2[i] = mean(d$df_orig[,2]$numpy())
  inter_mean_dgp_x3[i] = mean(d$df_orig[,3]$numpy())
  inter_dgp_x2[i,] = d$df_orig[,2]$numpy()
  inter_dgp_x3[i,] = d$df_orig[,3]$numpy()
}

#In 
df = data.frame(dox=numeric(0),x2=numeric(0),x3=numeric(0), type=character(0))
for (step in c(1,3,5,6,10)){
    df = rbind(df, data.frame(
      dox = dox_origs[step],
      x2 = inter_dgp_x2[step,],
      x3 = inter_dgp_x3[step,],
      type = 'simu'
    ))
    df = rbind(df, data.frame(
      dox = dox_origs[step],
      x2 = inter_ours_x2[step,],
      x3 = inter_ours_x3[step,],
      type = 'ours'
    )
  )
}
ggplot(df) + geom_density(aes(x=x2, col=type, linetype=type)) + facet_grid(~as.factor(dox))
ggplot(df) + geom_density(aes(x=x3, col=type, linetype=type)) + facet_grid(~as.factor(dox))

#X3
x1dat = data.frame(x=train$df_orig$numpy()[,1])
library(ggplot2)
df = data.frame(
  x = dox_origs,
  ours = inter_mean_ours_x3,
  #theoretical = dox_origs,
  Simulation_DGP = inter_mean_dgp_x3
) 
df %>% 
  pivot_longer(cols = 2:3) %>% 
  ggplot(aes(x=x, y=value, col=name, type=name)) + geom_point() +
  #geom_abline(intercept = 0, slope = 1, col='skyblue') +
  xlab("do(X1)") +
  ylab("E(X3|do(X1)") +
  geom_rug(data=x1dat, aes(x=x), inherit.aes = FALSE, alpha=0.5) +
  ggtitle(paste0('MSE ours vs Simulation: '))

#ATE from 0 to 1 (Hacky check that...)
dox_origs[7] #0
dox_origs[9] #1

ATE_our = df$ours[9] - df$ours[7] 
ATE_simu = df$Simulation_DGP[9] - df$Simulation_DGP[7] 
ATE_our - ATE_simu


########
#### Counterfact ####
###############################################
# Counterfactual  
###############################################

# CF of X4 given X1=alpha ###

#Creating a typical value 
if (FALSE){
  X1 = 1.
  x1 = scale_value(dat_train_orig = train$df_orig, col = 1, value = X1)$numpy()
  x2 = get_x(thetaNN_l[[2]], x1, 0.)
  x3 = get_x(thetaNN_l[[3]], c(x1,x2), 0.)
  df = data.frame(x1=0,x2=as.numeric(x2), x3=as.numeric(x3))
  unscaled = unscale(train$df_orig, tf$constant(as.matrix(df), dtype=tf$float32))$numpy()
  X2 = unscaled[2]
  X3 = unscaled[3]
  
  dat = val$df_orig$numpy()
  point = c(X1,X2,X3)
  distances <- apply(dat, 1, function(row) sqrt(sum((row - point)^2)))
  # Find the index of the closest row
  closest_row_index <- which.min(distances)
  # Retrieve the closest row
  closest_row <- dat[closest_row_index,]
  closest_row #1.081913  1.929156 18.40600
} else{
  X1 = 1.081913
  X2 = 1.929156
  X3 = 18.406000
}

###### CF Theoretical (assume we know the complete SCM) #####

#Abduction these are the Us that correspond the observed values
U1 = X1 - 1
U2 = X2 - 2*X1^2
U3 = X3 - 20./(1 + exp(-X2^2 + X1)) 

#X1 --> alpha
cf_do_x1_dgp = function(alpha){
  X_1 = alpha
  X_2 = 2*X_1^2 + U2 
  X_3 = 20./(1 + exp(-X_2^2 + X_1)) + U3
  return(data.frame(X1=X_1,X2=X_2,X3=X_3))
}

#Constistency
cf_dgp_cons = cf_do_x1_dgp(X1)
abs(cf_dgp_cons$X2-X2) #~1e-16 Consistency
abs(cf_dgp_cons$X3-X3) #~1e-16 Consistency
alpha = seq(-3,3,0.1)

###### CF From our model #####
# Scaling the observed data so that it can be used with the networks
cf_do_x1_ours = function(alpha){
  
  # Abduction (no need for alpha)
  ## Scaling of the observed variables
  x1 = scale_value(dat_train_orig = train$df_orig, col = 1, value = X1)$numpy()
  x2 = scale_value(dat_train_orig = train$df_orig, col = 2, value = X2)$numpy()
  x3 = scale_value(dat_train_orig = train$df_orig, col = 3, value = X3)$numpy()

  # Abduction Getting the relevant latent variable
  z2 =  get_z(thetaNN_l[[2]], x1, x2)
  z3 =  get_z(thetaNN_l[[3]], c(x1, x2), x3)

  ## Action
  a = scale_value(dat_train_orig = train$df_orig, col = 1, value = alpha)$numpy()
  x2_CF = get_x(net=thetaNN_l[[2]], parents = a, z2) #X1-->a but the same latent variable
  x3_CF = get_x(net=thetaNN_l[[3]], parents = c(a, x2_CF), z3) 

  df = data.frame(x1=x1, x2=as.numeric(x2_CF),x3=as.numeric(x3_CF))
  unscaled = unscale(train$df_orig, tf$constant(as.matrix(df), dtype=tf$float32))$numpy()
  return(unscaled)
}

#Consistency
cf_do_x1_dgp(X1) - cf_do_x1_ours(X1)

## Creating Results for do(x1)
df = data.frame()
for (a_org in c(seq(-3,3,0.2),X1)){
  print(a_org)
  dpg = cf_do_x1_dgp(a_org)
  df = bind_rows(df, data.frame(x1=a_org, X2=dpg[2], X3=dpg[3], type='DGP'))
  
  cf_our = cf_do_x1_ours(a_org)
  df = bind_rows(df, data.frame(x1=a_org, X2=cf_our[2], X3=cf_our[3], type='OURS'))
}

ggplot(df) +
  geom_point(data = subset(df, type == "OURS"), aes(x = x1, y = X2, color=type)) +
  geom_line(data = subset(df, type == "DGP"), aes(x = x1, y = X2, color=type)) + 
  xlab('would x1 be alpha')  
#

ggplot(df) +
  geom_point(data = subset(df, type == "OURS"), aes(x = x1, y = X3, color=type)) +
  geom_line(data = subset(df, type == "DGP"), aes(x = x1, y = X3, color=type)) + 
  xlab('would x1 be alpha')  
#

#####################
# Old code to check Do intervention
if (FALSE){
  dox1 = function(doX, thetaNN_l, num_samples){
    doX_tensor = doX * tf$ones(shape=c(num_samples,1L),dtype=tf$float32) 
    
    x2_samples = sample_from_target(thetaNN_l[[2]], doX_tensor)
    
    parents_x3 = tf$concat(list(doX_tensor, x2_samples), axis=1L)
    x3_samples = sample_from_target(thetaNN_l[[3]], parents_x3)
    
    return(matrix(c(doX_tensor$numpy(),x2_samples$numpy(), x3_samples$numpy()), ncol=3))
  }
  
  df = dox1(0.5, thetaNN_l, num_samples=1e4L)
  str(df)
  summary(df)
  
  df2t = do(thetaNN_l, train$A, doX=c(0.5, NA, NA), num_samples=1e4)
  df2 = as.matrix(df2t$numpy())
  qqplot(df2[,2], df[,2]);abline(0,1)
  qqplot(df2[,3], df[,3]);abline(0,1)
  qqplot(df2[,4], df[,4]);abline(0,1)
}









