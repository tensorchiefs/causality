# Extract the causal values from the DAG
get_causal_values <- function(dag_model){
  dag <- dag_model$dag
  # get edges
  x_from <- edges(dag)$v
  x_to <- edges(dag)$w
  
  # get the effects
  val <- c()
  for(i in 1:length(x_from)){
    f_args <- unlist(dag_model$f.args[x_to[i]])
    val[i] <- f_args[paste0(x_to[i],".betas.",x_from[i])]
  }
  # round values and rename
  val <- as.numeric(val)
  val <- round(val, digits = 3)
  names(val) <- paste0(x_from, " -> ", x_to)
  return(val)
}


# get adjacency matrix from dag
get_adj = function(dag, N) {
  A = matrix(0, ncol = N, nrow = N)
  # problems with no edges in dags in tidy_dagitty
  if(is.na(str_match(dag, "->")[1])){
    return(A)  
  }
  dd = tidy_dagitty(dag)$data
  A = matrix(0, ncol = N, nrow = N)
  for (i in 1:nrow(dd)) {
    from = as.integer(substring(dd$name[i],2))
    to = as.integer(substring(dd$to[i],2))
    A[from, to] = 1
  }
  return(A)
} 

get_adj_coef = function(pdm, N){
  A = get_adj(pdm$dag, N)
  Av = as.numeric(t(A))
  Av[which(Av == 1)] = get_causal_values(pdm)
  return(matrix(Av, nrow=N, byrow = TRUE))
} 


