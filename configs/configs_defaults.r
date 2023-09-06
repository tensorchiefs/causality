
DEBUG = FALSE     # prints debug messages
R_START = 1-0.0001    # left border for non-extrapolation of BP (orig [0,1])
L_START = 0.0001       # right border for non-extrapolation of BP (orig [0,1])
DEBUG_NO_EXTRA = FALSE  # does only use samples from latent, which are within h(LSTART) and h(RSTART) 
USE_EXTERNAL_DATA = TRUE  # we use external data instead of simulation from DGP or our model
EXTERNAL_DATA_FILE = "data/CAREFL_CF/X.csv"

# Repository URL
repo_url <- "https://github.com/tensorchiefs/causality"

latent_dist = tfd_logistic(loc=0, scale=1)  # distribution of the latent variable in TM


M = 30                  # order of BP
EPOCHS = 10000  
nTrain = 2500 

short_name = 'default'  # shot run Name 

get_suffix = function(short_name){
  sprintf("run_%s_M%d_E%d_nTrain%d", short_name, M, EPOCHS, nTrain)
}



DROPBOX = NA

seed = NA

coeffs = NA  # if DGP has coefs in SEM

dgp = function(){
  stop("dgp function not yet implemented")
}

optimizer= tf$keras$optimizers$Adam(learning_rate=0.001)