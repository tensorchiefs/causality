### Copy from 
# DB Causality_2022/tram_DAG/exp/carefl_eq8/runNormal_M30_carefldata_scaled

config_data <- yaml::yaml.load_file("R/tram_dag/Figure5.yaml")
DEBUG_NO_EXTRA = config_data$DEBUG_NO_EXTRA
normalized_config <- yaml::as.yaml(config_data)
config_id <- substring(openssl::sha256(charToRaw(normalized_config)),1,5)

git_headid_hash <- system("git rev-parse HEAD", intern = TRUE)
git_id = substring(git_headid_hash,1,8) #Needs to be 8 charaters long so that one can search on github.

repo_url <- "https://github.com/tensorchiefs/causality"
commit_url <- paste0(repo_url, "/commit/", git_id)
# Print the clickable URL in RStudio console
cat(commit_url)

# Check the status of the repository
git_status <- system("git status --porcelain", intern = TRUE)
# If there are uncommitted changes, stop execution
if (length(git_status) > 0) {
  stop("There are uncommitted changes in the Git repository. Please commit them before proceeding.")
}


#### Reproducing figure 5 with our code #####
library(R.utils)
source('R/tram_dag/utils.R')
DEBUG = FALSE
DEBUG_NO_EXTRA = FALSE
USE_EXTERNAL_DATA = TRUE

SUFFIX = 'runNormal_M30_carefldata_scaled'
M = 30
EPOCHS = 7000 
nTrain = 2500
DROPBOX = 'C:/Users/sick/dl Dropbox/beate sick/IDP_Projekte/DL_Projekte/shared_Oliver_Beate/Causality_2022/tram_DAG/'
DROPBOX = '~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/Causality_2022/tram_DAG/'
if (rstudioapi::isAvailable()) {
  context <- rstudioapi::getSourceEditorContext()
  this_file <- context$path
  print(this_file)
} else{
  this_file = "~/Documents/GitHub/causality/R/tram_dag/carefl_fig5.r"
}

#######################
# Latent Distribution
latent_dist = tfd_logistic(loc=0, scale=1)
#latent_dist = tfd_normal(loc=0, scale=1)
#latent_dist = tfd_truncated_normal(loc=0., scale=1.,low=-4,high = 4)
#hist(latent_dist$sample(1e5)$numpy(),100, freq = FALSE, main='Samples from Latent')

len_theta = M + 1
bp = make_bernp(len_theta)

######################################
############# DGP ###############
######################################
#coeffs <- runif(2, min = .1, max = .9)
coeffs = c(.5,.5)

# https://github.com/piomonti/carefl/blob/master/data/generate_synth_data.py

rlaplace <- function(n, location = 0, scale = 1) {
  p <- runif(n) - 0.5
  draws <- location - scale * sign(p) * log(1 - 2 * abs(p))
  return(draws)
}

#hist(rlaplace(10000),100)
#sd(rlaplace(10000))

dgp <- function(n_obs, coeffs, doX1=NA, dat_train=NULL, seed=NA, file=NULL) {
  if (is.na(seed) == FALSE){
    set.seed(seed)
  }
  
  #Use external data 
  if (is.null(file) == FALSE){
    data <- read.csv(file, header = FALSE)
    X_1 <- data[,1]
    X_2 <- data[,2]
    X_3 <- data[,3]
    X_4 <- data[,4]
    n_obs=length(X_4)
  } else{
    X <- matrix(rlaplace(2 * n_obs, 0, 1 / sqrt(2)), nrow = 2, ncol = n_obs)
    #X <- matrix(rnorm(2 * n_obs, 0, 1), nrow = 2, ncol = n_obs)
    X_1 <- X[1,]
    X_2 <- X[2,]
    
    if (is.na(doX1) == FALSE){
      X_1 = X_1 * 0 + doX1
    }
    
    X_3 <- X_1 + coeffs[1] * (X_2^3)
    X_4 <- -X_2 + coeffs[2] * (X_1^2)
    
    X_3 <- X_3 + rlaplace(n_obs, 0, 1 / sqrt(2))
    X_4 <- X_4 + rlaplace(n_obs, 0, 1 / sqrt(2))
    
    X_3 = X_3 / sd(X_3) 
    X_4 = X_4 / sd(X_4) 
}
  
  
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

#Data from CAREFL Fig 5
if (USE_EXTERNAL_DATA){
  file = "data/CAREFL_CF/X.csv"
  train = dgp(file = file, coeffs = coeffs, seed=42)
  #We compare against the results written in counterfactual_trials.py
  colMeans(train$df_orig$numpy()) #CAREFL [-0.01598893  0.00515872  0.00869695  0.26056851]
  apply(train$df_orig$numpy(), 2, sd) #CAREFL [1.03270839 0.98032533 1.         1.        ]
  X_obs <- read.csv('data/CAREFL_CF/xObs.csv', header = FALSE)[1,]
  X_obs = as.numeric(X_obs)
  X_obs #[[ 2.          1.5         0.84645028 -0.26158623]]
} 
pairs(train$df_orig$numpy())

#### Trying to reproduce the code in the paper #####
train_ours = dgp(nTrain, coeffs = coeffs, seed=42)
pairs(train_ours$df_orig$numpy())

qqplot(train_ours$df_orig$numpy()[,1], train$df_orig$numpy()[,1])
abline(0,1)
qqplot(train_ours$df_orig$numpy()[,2], train$df_orig$numpy()[,2])
abline(0,1)
qqplot(train_ours$df_orig$numpy()[,3], train$df_orig$numpy()[,3])
abline(0,1)
qqplot(train_ours$df_orig$numpy()[,4], train$df_orig$numpy()[,4])
abline(0,1)

library(igraph)
graph <- graph_from_adjacency_matrix(train$A, mode = "directed", diag = FALSE)
plot(graph, vertex.color = "lightblue", vertex.size = 30, edge.arrow.size = 0.5)

train_data = split_data(train$A, train$df_scaled)
thetaNN_l = make_thetaNN(train$A, train_data$parents)

if(USE_EXTERNAL_DATA){
  val = train
} else{
  val = dgp(5000, coeffs = coeffs, dat_train = train$df_orig)
}

val_data = split_data(val$A, val$df_scaled)


###### Training Step #####
optimizer= tf$keras$optimizers$Adam(learning_rate=0.001)
l = do_training(train$name, thetaNN_l = thetaNN_l, train_data = train_data, val_data = val_data,
                SUFFIX, epochs = EPOCHS,  optimizer=optimizer)

### Save Script
dirname = paste0(DROPBOX, "exp/", train$name, "/", SUFFIX, "/")
file.copy(this_file, dirname)

#e:200.000000  Train: -2.883684, Val: -2.179241 
loss = l[[1]]
loss_val = l[[2]]
plot(loss, type='l', ylim=c(-9.0,-1.0))
points(loss_val, col='green')

#Loading data from epoch e
load_weights(epoch = EPOCHS, l)


##########. Checking obs fit the marginals 
plot_obs_fit(train_data$parents, train_data$target, thetaNN_l, name='Training')
plot_obs_fit(val_data$parents, val_data$target, thetaNN_l, name='Validation')


############################### Do X via Flow ########################

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
  if (USE_EXTERNAL_DATA) {
    #The rescaling done in their code
    res_scm_x3[i] = res_scm_x3[i] / 6.01039
    res_scm_x4[i] = res_scm_x4[i]/ 1.9114155827
  }
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
  X3 = X_obs[3] / 6.01039440669
  X4 = X_obs[4] / 1.91141558279
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
    #if (USE_EXTERNAL_DATA){
    #  X_3 = X_3/6.01039440669
    #  X_4 = X_4/1.91141558279
    #}
    
   return(data.frame(X1=X_1,X2=X_2,X3=X_3,X4=X_4))
}

X_obs
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

xCF_onX1_true <- read.csv('data/CAREFL_CF/xCF_onX1_true.csv', header = FALSE)
df =  bind_rows(df, data.frame(x1=seq(-3,2.9,0.1), X2=NA, X3=NA, X4=xCF_onX1_true$V1, type='DGP_Code'))

x4tmp <- read.csv('data/CAREFL_CF/xCF_onX1_pred.csv', header = FALSE)
df =  bind_rows(df, data.frame(x1=seq(-3,2.9,0.1), X2=NA, X3=NA, X4=x4tmp$V1, type='CAREFL'))


x1dat = data.frame(x=train$df_orig$numpy()[,1])
ggplot(df) +
  geom_point(data = subset(df, type == "OURS"), aes(x = x1, y = X4, color=type)) +
  #geom_line(data = subset(df, type == "DGP"), aes(x = x1, y = X4, color=type)) + 
  geom_line(data = subset(df, type == "DGP_Code"), aes(x = x1, y = X4, color=type)) + 
  geom_point(data = subset(df, type == "CAREFL"), aes(x = x1, y = X4, color=type)) + 
  geom_rug(data=x1dat, aes(x=x), inherit.aes = FALSE, alpha=0.5) +
  xlab('would x1 be alpha')  

ggplot(df) +
  geom_point(data = subset(df, type == "OURS"), aes(x = x1, y = X3, color=type)) +
  geom_line(data = subset(df, type == "DGP"), aes(x = x1, y = X3, color=type)) + 
  geom_rug(data=data.frame(x=train$df_orig$numpy()[,1]), aes(x=x), inherit.aes = FALSE, alpha=0.5) +
  xlab('would x1 be alpha')  

df_would_x1_x4_cf = df
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

x3tmp <- read.csv('data/CAREFL_CF/xCF_onX2_true.csv', header = FALSE)
df =  bind_rows(df, data.frame(X1=NA, X2=seq(-3,2.9,0.1), X3=x3tmp$V1, X4=NA, type='DGP_Code'))

x3tmp <- read.csv('data/CAREFL_CF/xCF_onX2_pred.csv', header = FALSE)
df =  bind_rows(df, data.frame(X1=NA, X2=seq(-3,2.9,0.1),  X3=x3tmp$V1, X4=NA, type='CAREFL'))


ggplot(df) +
  geom_point(data = subset(df, type == "OURS"), aes(x = X2, y = X4, color=type)) +
  geom_line(data = subset(df, type == "DGP"), aes(x = X2, y = X4, color=type)) + 
  geom_rug(data=data.frame(x=train$df_orig$numpy()[,2]), aes(x=x), inherit.aes = FALSE, alpha=0.5) +
  xlab('would x2 be alpha')  

ggplot(df) +
  geom_point(data = subset(df, type == "OURS"), aes(x = X2, y = X3, color=type)) +
  #geom_line(data = subset(df, type == "DGP"), aes(x = X2, y = X3, color=type)) + 
  geom_line(data = subset(df, type == "DGP_Code"), aes(x = X2, y = X3, color=type)) + 
  geom_point(data = subset(df, type == "CAREFL"), aes(x = X2, y = X3, color=type)) + 
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











