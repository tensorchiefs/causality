##### TODO not finished see Protocol


library(reticulate)
EPOCHS = 10000
for (n_obs in c(250,500,750,1000,1250,1500,2000,2500)){
best_loss = 1e10
wait = 0
#n_obs = 2500

#### Reproducing figure 5 with our code #####
library(R.utils)
source('R/tram_dag/utils.R')
DEBUG = FALSE
DEBUG_NO_EXTRA = FALSE
USE_EXTERNAL_DATA = TRUE

optimizer= tf$keras$optimizers$Adam(learning_rate=0.001)
M = 30
nTrain = n_obs
DROPBOX = 'C:/Users/sick/dl Dropbox/beate sick/IDP_Projekte/DL_Projekte/shared_Oliver_Beate/Causality_2022/tram_DAG/'
DROPBOX = '~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/Causality_2022/tram_DAG/'
if (rstudioapi::isAvailable()) {
  context <- rstudioapi::getSourceEditorContext()
  this_file <- context$path
  print(this_file)
} else{
  this_file = "~/Documents/GitHub/causality/R/tram_dag/carefl_fig5.r"
}


SUFFIX = sprintf("runFig4_M%d_E%d_nTrain%d_Adam%.e", M, EPOCHS, nTrain,optimizer$lr$numpy())

############# Loading the experiment as run in python #######
load_pickle <- function(filepath) {

  
  reticulate::py_run_string(paste0("
import pickle
with open('", filepath, "', 'rb') as file:
    data = pickle.load(file)
  "))
  
  return(py$data)
}

data <- load_pickle(paste0('data/CAREFL_Fig4/int_', n_obs,'r_cl_mlp_5_10.p'))
head(data$X)
data$x3 #mse
data$x3e #mse stranfe
data$coef #Coefficients
data$loss #loss

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


dgp <- function(n_obs, coeffs, doX1=NA, dat_train=NULL, seed=NA, file=NULL, data=NULL) {
  if (is.na(seed) == FALSE){
    set.seed(seed)
  }
  
  #Use external data 
  if (is.null(data) == FALSE){
    X_1 = data$X[,1]
    X_2 = data$X[,2]
    X_3 = data$X[,3]
    X_4 = data$X[,4]
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
    
    X_3 = X_3 #/ sd(X_3) 
    X_4 = X_4 #/ sd(X_4) 
  }
  
  
  dat.s =  data.frame(x1 = X_1, x2 = X_2, x3 = X_3, x4 = X_4)
  dat.tf = tf$constant(as.matrix(dat.s), dtype = 'float32')
  A <- matrix(c(0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,0,0), nrow = 4, ncol = 4, byrow = TRUE)
  
  if (is.null(dat_train)){
    scaled = scale_df(dat.tf)
  } else{
    scaled = scale_validation(dat_train, dat.tf)
  }
  return(list(df_orig=dat.tf, df_scaled = scaled, coef=coeffs, A=A, name='carefl_eq8_no_scaling'))
} 

#Data from CAREFL Fig 4
if (USE_EXTERNAL_DATA){
  train = dgp(file = file, coeffs = NULL, seed=42, data=data)
  colMeans(train$df_orig$numpy()) #
  apply(train$df_orig$numpy(), 2, sd) #Not normalized as in the case of fig5
  
  #Comparing it with our DGP
  #array([0.45743466, 0.76959229])
  coeffs = c(0.3934242, 0.3380993)
  testing = dgp(coeffs = coeffs, seed=42, n_obs = n_obs)
  qqplot(train$df_orig$numpy()[,1], testing$df_orig$numpy()[,1]);abline(0,1)
  qqplot(train$df_orig$numpy()[,2], testing$df_orig$numpy()[,2]);abline(0,1)
  qqplot(train$df_orig$numpy()[,3], testing$df_orig$numpy()[,3]);abline(0,1)
  qqplot(train$df_orig$numpy()[,4], testing$df_orig$numpy()[,4]);abline(0,1)
} 

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

l = do_training(train$name, thetaNN_l = thetaNN_l, train_data = train_data, val_data = val_data,
                SUFFIX, epochs = EPOCHS,  optimizer=optimizer)

loss = l[[1]]
loss_val = l[[2]]

### Save Script
dirname = paste0(DROPBOX, "exp/", train$name, "/", SUFFIX, "/")
file.copy(this_file, dirname)

max_loss <- max(quantile(c(loss, loss_val), 0.9), data$loss)
data.frame(epochs = 1:length(loss), loss = loss, loss_val=loss_val) %>%
  ggplot(aes(x = epochs)) +
  geom_line(aes(y = loss, color = "Training Loss"), size = 1) +
  geom_line(aes(y = loss_val, color = "Validation Loss"), size = 1) +
  geom_hline(yintercept = data$loss, linetype = "dashed", color = "green") + 
  geom_text(aes(y = data$loss), x=0, label = "  loss CAREFL", vjust = 1, color = "green") +  # Label
  labs(x = "Epochs", y = "Loss", title = "Training and Validation Loss Over Epochs") +
  scale_color_manual(values = c("Training Loss" = "blue", "Validation Loss" = "red")) +
  theme_minimal() + 
  ylim(min(min(loss, data$loss)), max_loss) 

ggsave(make_fn("loss.pdf"))

}#Runs



#e:200.000000  Train: -2.883684, Val: -2.179241 
loss = l[[1]]
loss_val = l[[2]]

plot(loss, type='l', ylim=c(-8.0,-1.0))
points(loss_val, col='green')

#Loading model from epoch e
load_weights(epoch = EPOCHS, l)


#### Auswertung 
train_sizes = c(250,500,750,1000,1250,1500,2000,2500)
x3_mse_sample = x3_mse_med = rep(NA, length(train_sizes))
for (i in 1:length(train_sizes)){
  n_obs = train_sizes[i]
  SUFFIX = sprintf("runFig4_M%d_E%d_nTrain%d_Adam%.e", M, EPOCHS, nTrain,optimizer$lr$numpy())
  data <- load_pickle(paste0('data/CAREFL_Fig4/int_', n_obs,'r_cl_mlp_5_10.p'))
  head(data$X)
  data$x3 #mse
  data$x3e #mse stranfe
  data$coef #Coefficients
  #dox_origs = seq(-3, 2.95, by = 0.1) #np.arange(-3, 3, .1)
  x3_mse_sample[i] = data$x3
  x3_mse_med[i] = data$x3e
}
plot(train_sizes, x3_mse_sample)
lines(train_sizes, x3_mse_med)
plot(train_sizes, x3_mse_sample, log='y', ylim = c(1e-3,1e6), col='green')
lines(train_sizes, x3_mse_med, log='y', ylim = c(1e-3,1e6), col='red')

data <- load_pickle(paste0('data/CAREFL_Fig4/int_', n_obs,'r_cl_mlp_5_10.p'))
coeffs =  data$coef
#### Auswertung ####
dox_origs = seq(-3, 2.95, by = 0.2) #np.arange(-3, 3, .1)
num_samples = 200L

DEBUG_NO_EXTRA = TRUE
inter_mean_ours_x2 = inter_mean_ours_x3 = inter_mean_ours_x4 = NA*dox_origs
for (i in 1:length(dox_origs)){
  dox_orig = dox_origs[i]
  dox=scale_value(train$df_orig, col=1L, dox_orig)
  dat_do_x_s = do(thetaNN_l, A = train$A, doX = c(dox, NA, NA, NA), num_samples)
  
  df = unscale(train$df_orig, dat_do_x_s)
  
  inter_mean_ours_x2[i] = mean(df[,2]$numpy())
  inter_mean_ours_x3[i] = mean(df[,3]$numpy())
  inter_mean_ours_x4[i] = mean(df[,4]$numpy())
}

plot(dox_origs, inter_mean_ours_x3, xlim=c(0,5))

abline(0,1)

plot(dox_origs, inter_mean_ours_x4) 
lines(dox_origs, dox_origs^2*coeffs[2]) 

median((dox_origs - inter_mean_ours_x3)^2)
diff_x3 = diff_x4 = rep(NA, length(dox_origs))
for (i in 1:length(dox_origs)){
  dox_orig = dox_origs[i]
  diff_x3[i] = (inter_mean_ours_x3[i] - dox_orig)
  diff_x4[i] = (inter_mean_ours_x4[i] - dox_orig^2*coeffs[2])
}
plot(dox_origs, diff_x3)
plot(dox_origs, diff_x4)
median(diff_x3^2) #TODO Fix outlier and change to mean
median(diff_x4^2) 
mean(diff_x4^2) 


