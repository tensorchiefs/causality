#### Data Simultion 
# We simmulate an identifiable causal graph with a hidden confounder
# x1 -> x2 -> x3
# The hidden confounder will act on both x2 and x3

N = 50000
flip = sample(c(0,1), N, replace = TRUE)
h = rnorm(N, 0, 0.5)
x1 = h + flip*rnorm(N, -2, sqrt(1.5)) + (1-flip) * rnorm(N, 1.5, 1)
x2 = 0.42*x1 +  rnorm(N, 0,1.42)
x3 = x2 + 2*h + rnorm(N,0,0.5)


#Estimation direct causal effect x1 -> x2 (should be 0.42)
confint(lm(x2 ~ x1))

#Estimation direct causal effect x2 -> x3 (should be 1)
confint(lm(x3 ~ x2))
# This does not work because of the hidden confounder

# Adjust for the hidden confounder h to block the backdoor path between x2 and x3 
confint(lm(x3 ~ x2 + h)) 
# This is a bit cheating because we dont know the true value of h

# Block the backdoor path x2-> x3
confint(lm(x3 ~ x2 + x1)) 

# Using stan
library(cmdstanr)
m_rcmdstan <- cmdstan_model('R/hidden_confounder_lin.stan')
m_rcmdstan$optimize(data = list(N=N, x1=x1,x2=x2,x3=x3))
s_rcmdstan

options(mc.cores = parallel::detectCores())
s_rcmdstan = m_rcmdstan$sample(data = list(N=N, x1=x1,x2=x2,x3=x3))
s_rcmdstan


