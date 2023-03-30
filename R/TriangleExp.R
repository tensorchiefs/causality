library(reticulate)
np <- import("numpy")
X <- np$load("/Users/oli/Documents/workspace_other/VACA/VACA/triangle_linear/train_5000_X.npy")
#U <- np$load("/Users/oli/Documents/workspace_other/VACA/VACA/triangle_linear/train_5000_U.npy")


hist(X[,1], 100)

#Vaca Paper 
#Table 5,6: Structural equations
#MoG(0.5N (−2, 1.5) + 0.5N (1.5, 1))
## 1.5 is variance!
N = 5000
flip = sample(c(0,1), N, replace = TRUE)
x1 = flip*rnorm(N, -2, sqrt(1.5)) + (1-flip) * rnorm(N, 1.5, 1)
x2 = -x1 + rnorm(N, 0,1)
x3 = x1 + 0.25*x2 + rnorm(N,0,1)

#Estimation direct causal effect x1 -> x3 (should be a31 = 1)
confint(lm(x3 ~ x1 + x2)) #Need to block x1->x2->x3 (open mediator)

#Estimation direct causal effect x1 -> x2 (should be a21 = -1)
confint(lm(x2 ~ x1)) #Don't Adjust for the collider x3

#Estimation direct causal effect x2 -> x3 (should be a32 = 0.25)
confint(lm(x3 ~ x2 + x1)) #Adjust for the confounder x1
library(cmdstanr)
m_rcmdstan <- cmdstan_model('R/triangle_lin.stan')
s_rcmdstan = m_rcmdstan$sample(data = list(N=N, x1=x1,x2=x2,x3=x3))
library(tidybayes)
s_rcmdstan


##########################
# Nonlinear
#MoG(0.5N (−2, 1.5) + 0.5N (1.5, 1))
#
options(mc.cores = parallel::detectCores())
x2 = -1 + 3 / (1 + exp(-2*x1)) + rnorm(N,0,sqrt(0.1))
x3 = x1 + 0.25*(x2)^2 + rnorm(N,0,1)
nlin_model <- cmdstan_model('R/triangle_nlin.stan')
nlin_samples = nlin_model$sample(data = list(N=N, x1=x1,x2=x2,x3=x3))


nlin_model$optimize(data = list(N=N, x1=x1,x2=x2,x3=x3))



