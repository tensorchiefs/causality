library(deepregression)
library(deeptrafo)
library(mlt)
library(tram)
library(MASS)
library(tensorflow)
library(tidyverse)

library("deeptrafo")

dgp <- function(n) {
  x <- rnorm(n)
  y <- x + rnorm(n)
  z <- 1.5*y + 0.42*x + rnorm(n)
  data.frame(x = x, y = y, z = z)
}

dat <- dgp(42)
m1 <- ColrNN(x ~ 1, data = dat) #Colr --> f_z is dlogis

#### Calculation of the NLL
## It's tricky
y_preds = fm1 = fitted(m1)
# 4 columns: 
##  shift which is beta*x_i in our case
##  h(y_upper) which is Base h(y_i) in case of a continous y (and no complex intercept)  
##. h(y_lower) not needed in our case
##  h'(y_upper) is h'(y_i)
## The complete transformation
##    h(y_i|x_i) = h(y_i) + shift
logLik(m1,dat, convert_fun = identity)
-log(dlogis(fm1[,1]+fm1[,2])*fm1[,4])

# fit(m1, epochs = 0)
logLik(m1,dat, convert_fun = mean)
## [1] -196.2493
# 1.962493  bigNLL  --> passt
mean(-log(dlogis(fm1[,1]+fm1[,2])*fm1[,4]))


m2 <- ColrNN(y ~ x, data = dat)
# fit(m2, epochs = 0)
# logLik(m2,dat)
# # [1] -511.9449
# 
# # both manually -708.1942
# logLik(m1,dat)+logLik(m2,dat)  # -708.1942
# # 7.0819416 # bigNLL  --> passt

m3 <- ColrNN(z ~ y + x, data = dat)

#Constructs the loss fkt of the combined model
get_loss <- function(mods) {
  function(y_true, y_pred) {
    #Agregating the NLL over the observatios for each model
    nlls <- lapply(seq_along(mods), 
                   function(mod) {
                        k_mean(layer_add(mods[[mod]]$model$loss(y_true[[mod]], y_pred[[mod]])))
                    }
                   )
    k_sum(nlls) #Aggregates the NLL of the models  #TODO check mean/sum
  }
}


rename = function(m, offset){
  for(layer in m$model$layers)
    layer$`_name` = paste0(layer$name, offset)
  return (m)
} 

m2 = rename(m2, '_m2')
m1 = rename(m1, '_m1')
m3 = rename(m1, '_m3')
m1$model
m1$model$input


mods <- list(m1, m2, m3)
# mods = list(m1, m2)
#mods= list(m1)

y_trues <- lapply(mods, \(x) x$init_params$y) # just indicator für censor type
y_preds <- lapply(mods, fitted)

m1$model$loss(y_trues[[1]], y_preds[[1]]) #The loss contr
# determine loss over all nodes
big_nll <- get_loss(mods)   #The loss fkt for the combined model
big_nll(y_trues, y_preds)   # stimmt, aber wie genau berechnet in fct?

IF (FALSE){
  #Checking the NLL for an untrained model
  y_new = predict(m2, dat)
  dat_new = data.frame(x=dat$x, y=y_new, z=dat$z)
  m3 <- ColrNN(z ~ y + x, data = dat_new)
  mods <- list(m1, m2, m3)
  logLik(m1,dat, convert_fun = mean) +
    logLik(m2,dat, convert_fun = mean) +
    logLik(m3,dat_new, convert_fun = mean)
}

inp <- list(m1$model$input, m2$model$input)#, m3$model$input)  #TODO Output
out <- list(m1$model$output)#, m2$model$output, m3$model$output)
big_mod <- keras_model(inp, out)
compile(big_mod, loss = get_loss(mods))

# input represents the trafo
# for m1 we have 5 elements for the baselin-trafo
# for m2 and m3 we have 6 elments: 5 for baseline, one for shift in trafo

# output has 4 cols per model, holding output for different censor types

fit(big_mod)
