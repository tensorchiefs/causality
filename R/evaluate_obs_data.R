### Package loading ######
library(dagitty)
library(ggdag) #dagitty_to_adjmatrix(d)
library(tidyr)
library(ggplot2)
library(tidyverse)
library(keras)
library(bnlearn)
library(igraph)
library(pcalg)
library("Rgraphviz")

# Loading of the adajacency matrix from the experiments
# 
load('~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/Causality_2022/R/DAG_Learning/data/3_okt_22/test_adj.rda')
rm(adj_test_pc) #has been wrong (pmax instead of ...)
n_test = nrow(adj_test_true)
load('~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/Causality_2022/R/DAG_Learning/data/3_okt_22/test_pcalg.rda')

dumm_opt = function(prob1){# = 0.2 * 25/20){
  dumm_geraten = matrix(0, nrow = nrow(adj_test_nn), ncol = ncol(adj_test_nn))
  prob1  #More than 20 percent, since 1's in diagonal will be removed
  for (i in 1:nrow(dumm_geraten)){
    d = sample(c(0,1), size=25, prob = c(1-prob1, prob1), replace = TRUE)
    #d[c(1,6,10,15,20,25)] = 0 #Not in diagnoal
    dumm_geraten[i,] = d
  }
  test_rmse_dumm = rep(NA, n_test)
  for (i in 1:n_test){
    test_rmse_dumm[i] = sqrt(mean(((1.0*dumm_geraten - adj_test_true[i,])**2)))
  }
  return(list(r=test_rmse_dumm, adj=dumm_geraten))
}

res = ps = seq(0,0.5, 0.05)
for (i in 1:length(ps)){
  d = dumm_opt(ps[i])
  res[i] = mean(d$r)
}
plot(ps, res)
abline(h=sqrt(5/25))
sqrt(5/25)
rmse_all_zero = mean(sqrt(rowMeans(adj_test_true[1:100,])))
abline(h = rmse_all_zero)

#Quick Hack
#adj_test_pc = dumm_geraten


#### ROC Analysis to find cutoff #####
score_nn_edge = adj_test_nn[adj_test_true > 0.5] #adj_test_true > 0 ==> edge
score_nn_noedge = adj_test_nn[adj_test_true <= 0.5]
plot(density(score_nn_edge), col='green', xlab = 'score', main = 'green:edge, blue:noedge')
lines(density(score_nn_noedge),col='blue')
hist(score_nn_noedge)
hist(score_nn_edge)

abline(v=0.1)
library(ROCR)
nn <- prediction(as.vector(adj_test_nn), as.vector(adj_test_true))
plot(performance(nn, "tpr", "fpr"))
abline(0,1)
performance(nn,"auc")@y.values

### optimal cut-off (using 100 examples)
options('digits'=3)
get_hamming = function(cut_adj) {
  test_hamming_nn = test_hamming_pc = rep(NA, 100)
  for (i in 1:100){
    adj_nn = 1.0*(adj_test_nn[i,] > cut_adj)
    test_hamming_nn[i] = sum(abs(1.0*adj_nn - adj_test_true[i,]))
    test_rmse_nn[i] = sqrt(mean(((1.0*adj_nn - adj_test_true[i,])**2)))
  }
  return (list(ham = mean(test_hamming_nn), rmse = mean(test_rmse_nn)))
}

res_h = res_r = cuts = seq(0,1,0.01)
for (i in 1:length(cuts)){
  d = get_hamming(cuts[i])
  res_h[i] = d$ham
  res_r[i] = d$rmse
}
plot(cuts, res_h)
plot(cuts, res_r)

### Calculation of RMSE and Hamming using the Adj Matrix
cut_adj = 0.3
test_rmse_nn = test_rmse_pc = test_hamming_nn = test_hamming_pc = rep(NA, n_test)
for (i in 1:n_test){
  adj_pc = adj_test_pc[i,] # adjacency matrix from pcalg object
  adj_nn = 1.0*(adj_test_nn[i,] > cut_adj)
  test_hamming_nn[i] = sum(abs(1.0*adj_nn - adj_test_true[i,]))
  test_hamming_pc[i] = sum(abs(1.0*adj_pc - adj_test_true[i,]))
  test_rmse_nn[i] = sqrt(mean(((1.0*adj_nn - adj_test_true[i,])**2)))
  test_rmse_pc[i] = sqrt(mean(((1.0*adj_pc - adj_test_true[i,])**2)))
}
hist(test_hamming_nn)
hist(test_hamming_pc)
table(test_hamming_nn)
table(test_hamming_pc)
summary(test_hamming_nn)
summary(test_hamming_pc)
summary(test_rmse_nn)
summary(test_rmse_pc)
hist(test_rmse_nn)
hist(test_rmse_pc)

#### Filtering of V-Structures (Adjecency Matrix gets the value of 2)
N = as.integer(sqrt(ncol(adj_test_true)))
adj_test_true_v = adj_test_true #Edges involved in a v-structure are coded with 2
c_v = 0 #Number of graphs with at least one v-structure
for (i in 1:nrow(adj_test_true)){
  a = matrix(adj_test_true[i,], ncol=N)
  #plot(graph_from_adjacency_matrix(a)) #Compated against first testdag from `290222_dag_sim.rda`
  a[,colSums(a) > 1] = a[,colSums(a) > 1] * 2 
  adj_test_true_v[i,] = matrix(a, nrow=1)
  if(max(colSums(a) > 1)){
    c_v = c_v +1
  }
}
c_v
mean(adj_test_true_v == 2) #proportion of possible edges which are in v-structure 
mean(adj_test_true_v == 1) #proportion of possible edges not involved in v-structure
mean(adj_test_true_v == 0) #proportion of non-edges

mean(adj_test_pc[adj_test_true_v == 2] == 1) #0.8126536
mean(adj_test_pc[adj_test_true_v == 1] == 1) #0.9560724
mean(adj_test_pc[adj_test_true_v == 0] == 0) #0.8745296
mean(adj_test_pc[adj_test_true_v == 0] == 1) #0.1254704

cut_adj = 0.3#0.1
mean(adj_test_nn[adj_test_true_v == 2] > cut_adj) #0.985
mean(adj_test_nn[adj_test_true_v == 1] > cut_adj) #1
mean(adj_test_nn[adj_test_true_v == 0] < cut_adj) #0.744



