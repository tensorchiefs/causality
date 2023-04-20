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
  z <- y + rnorm(n)
  data.frame(x = x, y = y, z = z)
}

dat <- dgp(1e2)
m1 <- ColrNN(x ~ 1, data = dat)

# fit(m1, epochs = 0)
# logLik(m1,dat)
## [1] -196.2493
# 1.962493  bigNLL  --> passt

logLik(m1,dat, convert_fun=identity)

m2 <- ColrNN(y ~ x, data = dat)

# fit(m2, epochs = 0)
# logLik(m2,dat)
# # [1] -511.9449
# 
# # both manually -708.1942
# logLik(m1,dat)+logLik(m2,dat)  # -708.1942
# # 7.0819416 # bigNLL  --> passt


m3 <- ColrNN(z ~ y, data = dat)

fitted(m1) # 4 columns: shift h(y_upper) h(y_lower) h'(y_upper)

m1$init_params$y # censoring indicators
m1$init_params$response # actual responses

get_loss <- function(mods) {
  big_loss <- function(y_true, y_pred) {
    nlls <- lapply(seq_along(mods), \(mod) {
      k_mean(layer_add(mods[[mod]]$model$loss(y_true[[mod]], y_pred[[mod]])))
    })
    k_sum(nlls)
  }
}



mods <- list(m1, m2, m3)
# mods = list(m1, m2)
#mods= list(m1)

big_nll <- get_loss(mods)

y_trues <- lapply(mods, \(x) x$init_params$y) # indicator für censor type

y_preds <- lapply(mods, fitted)

# determine loss over all nodes
big_nll(y_trues, y_preds)   # stimmt, aber wie genau berechnet in fct?

# determine individual logLik contributions
(m1$model$loss(m1$init_params$y, fitted(m1))) 

# input represents the trafo
# for m1 we have 5 elements for the baselin-trafo
# for m2 and m3 we have 6 elments: 5 for baseline, one for shift in trafo
inp <- list(m1$model$input, m2$model$input, m3$model$input) 

# output has 4 cols per model, holding output for different censor types
out <- list(m1$model$output, m2$model$output, m3$model$output)

big_mod <- keras_model(inp, out)
compile(big_mod, loss = get_loss(mods))
fit(big_mod)
