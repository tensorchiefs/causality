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
var(X) #In DAG: X1->X2 but the variance of X1 is higher than X2
varsortability(X, get_adj(g, N))
sortnregress(X)
get_adj(g, N)

### parameter setting ###############
N=4  # number nodes
p=0.5  # connectivity
dag = randomDAG(N, p)
str(dag)
plot(dag)
get_adj(dag, N)

if (FALSE){
  pdm <- parametric_dag_model(dag, 
                              f.args = list(x2 = list(betas = list(x1 = 1.1)),
                                            x3 = list(betas = list(x1 = -2, x2=0.5))
                                            ))
}
pdm = parametric_dag_model(dag)
get_causal_values(pdm)

#Simulation of observables from the parameteric dag model
sim_data <- sim_mixed_dag(dag_model = pdm, N = 100)
confint(lm(x2 ~ x1, sim_data))
round(var(sim_data),2)
varsortability(sim_data, get_adj(dag, N))
sortnregress(as.matrix(sim_data))

# test sortnregress
# Example from rom: https://github.com/Scriddie/Varsortability/blob/main/src/sortnregress.py
#
#W = np.array([[0, 1, 0], [0, 0, 2], [0, 0, 0]])
#X = np.random.randn(1000, 3).dot(np.linalg.inv(np.eye(3) - W))
W <- matrix(c(0, 1, 0, 0, 0, 2, 0, 0, 0), nrow = 3, ncol = 3, byrow = TRUE)
X = matrix(rnorm(1000*3), ncol=3)
X <- X %*% solve(diag(3) - W)
sortnregress(X)
# Python Reference implementation (using BIC)
# [[0.         1.01499348 0.        ]
#  [0.         0.         1.96755702]
#  [0.         0.         0.        ]]


#### DAG with no tear
#Understanding of the loss function as trace(exp(W^2)-d)

DAG = sign(W)
DAG**2
DAG %*% DAG
DAG %*% DAG %*% DAG

B %*% B %*% B
B = W^2 #Makes W positive
(expm(B)) #Trace --> 3 (3 ones on the diagonal)
(expm(100*B)) #Trace --> 3 (Still 3 ones on the diagonal) 3 is number of Variables (dim of Adjecency)

C = matrix(c(0,1,0,1,0,0,0,0,0), nrow = 3, byrow = TRUE)
C %*% C %*% C %*% C
expm(C^2) #Trace > 3
expm(100*C^2) #Trace > 3 (has influces)



