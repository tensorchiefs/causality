library(keras)
library(tensorflow)
source('R/tram_dag/utils_dag_maf.R') #Might be called twice
source('R/tram_dag/utils.R') #L_START

### Defining the DGP
dgp <- function(n_obs) {
    print("=== Using the DGP of the VACA1 paper in the linear Fashion (Tables 5/6)")
    flip = sample(c(0,1), n_obs, replace = TRUE)
    X_1 = flip*rnorm(n_obs, -2, sqrt(1.5)) + (1-flip) * rnorm(n_obs, 1.5, 1)
    X_2 = -X_1 + rnorm(n_obs)
    X_3 = X_1 + 0.25 * X_2 + rnorm(n_obs)
    dat.s =  data.frame(x1 = X_1, x2 = X_2, x3 = X_3)
    dat.tf = tf$constant(as.matrix(dat.s), dtype = 'float32')
    scaled = scale_df(dat.tf) * 0.99 + 0.005
    A <- matrix(c(0, 1, 1, 0,0,1,0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
    return(list(df_orig=dat.tf,  df_scaled = scaled, A=A, dpg_name="VACA1"))
} 

##### Parameters
N_train = 50000
train = dgp(N_train)
dpg_name = train$dpg_name
hidden_features = c(2,2)
adjacency <- t(train$A)
layer_sizes <- c(ncol(adjacency), hidden_features, nrow(adjacency))
M = 30L
Epochs = 100L
### Make Unique Filename for the model using dpg_name, M, epochs, hidden_features
filename = paste0(dpg_name, "_M", M, "_N",N_train, "_E", Epochs, "_", paste0(hidden_features, collapse = "_"), ".h5")
filename
#filename = VACA1_M30_N50000_E100_2_2_TrainedPython.h5


masks = create_masks(adjacency = adjacency, hidden_features)
param_model = create_theta_tilde_maf(adjacency = adjacency, len_theta = M+1, layer_sizes = layer_sizes)
optimizer = optimizer_adam()
param_model$compile(optimizer, loss=dag_loss)
param_model$evaluate(x = train$df_scaled, y=train$df_scaled, batch_size = 100L)

##### Training or loading ####
if (file.exists(filename)){
  param_model$load_weights(filename)
  #param_model$load_weights("triangle_test_large_long.keras_weights.h5")
} else {
  hist = param_model$fit(x = train$df_scaled, y=train$df_scaled, epochs = Epochs,verbose = TRUE)
  param_model$save_weights(filename)
  plot(hist$epoch, hist$history$loss)
}

param_model$evaluate(x = train$df_scaled, y=train$df_scaled, batch_size = 100L)
s = do_dag(param_model, train$A, doX=c(NA, NA, NA))
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("X_",i))
  lines(density(train$df_scaled$numpy()[,i]))
}

s = do_dag(param_model, train$A, doX=c(0.5, NA, NA))
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("Do (X3=0.5) X_",i))
  lines(density(train$df_scaled$numpy()[,i]))
}




















