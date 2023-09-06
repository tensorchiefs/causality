# define parameters different than in cofigs_default.r

# global Variable

USE_EXTERNAL_DATA = TRUE  # we use external data instead of simulation from DGP or our model
EXTERNAL_DATA_FILE = "data/CAREFL_CF/X.csv"


# DGP

# noise for dgp
rlaplace <- function(n, location = 0, scale = 1) {
  p <- runif(n) - 0.5
  draws <- location - scale * sign(p) * log(1 - 2 * abs(p))
  return(draws)
}

coeffs = c(0.5,0.5)  #  coefs of SEM in DGP

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
    #X_3 <- X_3 + rnorm(n_obs, 0, 1 )
    #X_4 <- X_4 + rnorm(n_obs, 0, 1 )
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

