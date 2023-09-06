DEBUG = TRUE
source('tram_scm/bern_utils.R')
library(R.utils)
library(tfprobability)

# Evaluation of to_theta
theta_im <- tf$constant(matrix(c(1,2,3,4,5,6.6), byrow = TRUE, ncol = 3), shape=c(2L,3L), dtype = 'float32')
theta_im = to_theta(theta_im)

# Evaluation of eval_h
y = tf$constant(c(0.1,0.2), shape=c(2L,1L))
d = bernp(3)
eval_h(theta_im, y, d$beta_dist_h)$numpy()

# Evaluation of eval_h
eval_h_dash(theta_im, y, d$beta_dist_h_dash)$numpy()


##### Testing the extrapolation ######
if (FALSE){
  d = bernp(4)
  # Test input data
  ys = seq(-1,2,0.01)
  h = NA*ys
  hextra = h 
  hdash = h
  for (i in 1:length(ys)){
    theta_im <- tf$constant(c(-0.1, 12.2, 22.34, 122.4), shape=c(1L,4L))
    y_i <- tf$constant(ys[i], shape=c(1L,1L))
    # Call eval_h_inter function
    h[i] = eval_h(theta_im, y_i, d$beta_dist_h)$numpy()
    hdash[i] = eval_h_dash(theta_im, y_i, d$beta_dist_h_dash)$numpy()
    hextra[i]=eval_h_extra(theta_im, y_i, d$beta_dist_h, d$beta_dist_h_dash)$numpy()
  }
  plot(ys, hextra, col='red', lty=2, type='l')
  lines(ys, h,  type='l')
  #lines(ys, hdash, col='pink', lty=2)
  
  d$beta_dist_h$prob(y_i)
  eval_h
  
  y_i <- tf$constant(c(0.1,1.01), shape=c(2L,1L))
  eval_h(theta_im, y_i, d$beta_dist_h)
  eval_h_extra(theta_im, y_i, d$beta_dist_h, d$beta_dist_h_dash)
}
