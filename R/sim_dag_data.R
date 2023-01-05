### Library ######
library(dagitty)
library(tidyverse)
library(pacman)
#library(help="pacman")
library(simMixedDAG)
source('R/causality_utils.R')

set.seed(1234)
### parameter setting ###############

N=5  # number nodes
p= 0.5  # connectivity
n_obs = 5000  # number of obs simulated from DAG's SEM
k_train = 2000  # number of DAGs
k_test = 500

### Probelauf #########################

dag = randomDAG(N, p)
str(dag)
plot(dag)
get_adj(dag, N)

# specify from DAG structure a paramteric DAG models (SEM)
# second argiment f.args let to default
param_dag_model <- parametric_dag_model(dag)
get_adj_coef(param_dag_model, N)
get_causal_values(param_dag_model)

# from parametric DAG (SEM) we can simulate data
sim_data <- sim_mixed_dag(dag_model = param_dag_model, 
                          N =n_obs)
ggplot(sim_data,aes(x2, x1)) + geom_point() +
  stat_smooth(method = "lm")
fit=lm(x2 ~ x1, data=sim_data)
summary(fit)
confint(fit)

### simulate train and test DAGs ####################

train_dags = list()
for (k in 1:k_train){
  train_dags[[k]] = randomDAG(N, p)
}

test_dags = list()
k=1
while (k <= k_test){
  dag = randomDAG(N, p)
  if( dag %in% train_dags == FALSE ) {
     test_dags[[k]] = dag
     k = k+1
  }
}

### simulate observations from train and test DAGs #########

train_obs = list()
for (i in 1:length(train_dags)){
  dag_model = parametric_dag_model(dag = train_dags[[i]])
  train_obs[[i]] = sim_mixed_dag(dag_model, n_obs)
}

test_obs = list()
for (i in 1:length(test_dags)){
  dag_model = parametric_dag_model(dag = test_dags[[i]])
  test_obs[[i]] = sim_mixed_dag(dag_model, n_obs)
}



