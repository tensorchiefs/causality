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
  SDX3 = 6.01039440669
  SDX4 = 1.91141558279
} 


df = train$df_orig$numpy()
df = rbind(df, X_obs)
pairs(df, col=c(rep('black', nrow(df)-1), 'red'), cex=c(rep(1, nrow(df)-1),2))


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


###############################################
##### Counterfactual  #########################
###############################################
### CF of X4 given X1=alpha ####

X1 = X_obs[1]
X2 = X_obs[2]
X3 = X_obs[3] 
X4 = X_obs[4] 

cf_do_x1_dgp = function(alpha){
    ###### Theoretical (assume we know the complete SCM)
    # X_3 <- (X_1 + coeffs[1] * (X_2^3)  + U3)/SDX3
    # X_4 <- (-X_2 + coeffs[2] * (X_1^2) + U4)/SDX4
    
    U1 = X1
    U2 = X2
    U3 = X3*SDX3 - X1 - coeffs[1] * X2^3
    U4 = X4*SDX4 - coeffs[2]*X1^2 + X2
    #X1 --> alpha
    X_1 = alpha
    X_2 = U2
    X_3 = (X_1 + coeffs[1] * (X_2^3)  + U3)/SDX3
    X_4 = (-X_2 + coeffs[2] * (X_1^2) + U4)/SDX4
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
for (a_org in c(seq(-3.5,3.5,0.05),X1)){
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

### Loading Ground Thruth from Data
xCF_onX1_true <- read.csv('data/CAREFL_CF/xCF_onX1_true.csv', header = FALSE)
df =  bind_rows(df, data.frame(x1=seq(-3,2.9,0.1), X2=NA, X3=NA, X4=xCF_onX1_true$V1, type='DGP_Code'))

x4tmp <- read.csv('data/CAREFL_CF/xCF_onX1_pred.csv', header = FALSE)
df =  bind_rows(df, data.frame(x1=seq(-3,2.9,0.1), X2=NA, X3=NA, X4=x4tmp$V1, type='CAREFL'))


x1dat = data.frame(x=train$df_orig$numpy()[,1])
ggplot(df) +
  geom_point(data = subset(df, type == "OURS"), aes(x = x1, y = X4, color=type)) +
  geom_line(data = subset(df, type == "DGP"), aes(x = x1, y = X4, color=type)) + 
  #geom_line(data = subset(df, type == "DGP_Code"), aes(x = x1, y = X4, color=type)) + 
  geom_point(data = subset(df, type == "CAREFL"), aes(x = x1, y = X4, color=type)) + 
  geom_rug(data=x1dat, aes(x=x), inherit.aes = FALSE, alpha=0.5) +
  xlab('would x1 be alpha')  

ggsave('~/Dropbox/Apps/Overleaf/tramdag/figures/carefel_fig5_left.pdf', width = 6, height=6)


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
  # X_3 <- (X_1 + coeffs[1] * (X_2^3)  + U3)/SDX3
  # X_4 <- (-X_2 + coeffs[2] * (X_1^2) + U4)/SDX4
  U1 = X1
  U2 = X2
  U3 = X3*SDX3 - X1 - coeffs[1] * X2^3
  U4 = X4*SDX4 - coeffs[2]*X1^2 + X2
  #X1 --> alpha
  X_1 = U1
  X_2 = alpha
  X_3 = (X_1 + coeffs[1] * (X_2^3)  + U3)/SDX3
  X_4 = (-X_2 + coeffs[2] * (X_1^2) + U4)/SDX4
  return(data.frame(X1=X_1,X2=X_2,X3=X_3,X4=X_4))
}

abs(cf_do_x2_dgp(X2)-X_obs) #~1e-16 Consistency

## Creating Results for do(x2)
df = data.frame()
for (a_org in c(seq(-5,5,0.05),X2)){
  dgp = cf_do_x2_dgp(a_org)
  df = bind_rows(df, data.frame(x1=dgp[1], X2=a_org, X3=dgp[3], X4=dgp[4], type='DGP'))
}

for (a_org in c(seq(-4.5,4.5,0.5),X2)){
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
  geom_line(data = subset(df, type == "DGP"), aes(x = X2, y = X3, color=type)) + 
  geom_line(data = subset(df, type == "DGP_Code"), aes(x = X2, y = X3, color=type)) + 
  geom_point(data = subset(df, type == "CAREFL"), aes(x = X2, y = X3, color=type)) + 
  geom_rug(data=data.frame(x=train$df_orig$numpy()[,2]), aes(x=x), inherit.aes = FALSE, alpha=0.5) +
  xlab('would x2 be alpha')  

ggsave('~/Dropbox/Apps/Overleaf/tramdag/figures/carefel_fig5_right.pdf', width = 6, height=6)






