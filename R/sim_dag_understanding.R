### Library ######
library(dagitty)
library(tidyverse)
library(pacman)
#library(help="pacman")
library(simMixedDAG)
source('R/causality_utils.R')

### Example from Reisach paper chapter 3.1
randomDAG(3, 0.3)
g = dagitty(
  "dag {
    x1
    x2
    x3
    x3 <- x1 -> x2 -> x3
  }"
)
plot(g)
n = 1000
N = 3
x1 = rnorm(n, mean=0, sd=sqrt(2)) #var 2
x2 = 0.5*x1 + rnorm(n, mean=0, sd=sqrt(1 - 0.5^2 * var(x1)))
varx2 = 1
varx1 = 2
var3 = 0.5^2*varx1 + 0.25^2*varx2 + 2*0.5*0.25*cov(x1,x2)
x3 = 0.5*x1 + 0.25*x2 +  rnorm(n, mean=0, sd=sqrt(3 - var3))
var(x3)
X = as.matrix(data.frame(x1=x1,x2=x2, x3=x3))
varsortability(X, get_adj(g, N))

### parameter setting ###############
set.seed(4)
N=4  # number nodes
p=0.5  # connectivity
dag = randomDAG(N, p)
str(dag)
plot(dag)
get_adj(dag, N)

pdm <- parametric_dag_model(dag, 
                            f.args = list(x2 = list(betas = list(x1 = 1.1)),
                                          x3 = list(betas = list(x1 = -2, x2=0.5))
                                          ))
pdm = parametric_dag_model(dag)
get_causal_values(pdm)

#Simulation of observables from the parameteric dag model
sim_data <- sim_mixed_dag(dag_model = pdm, N = 100)
confint(lm(x2 ~ x1, sim_data))




