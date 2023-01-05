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