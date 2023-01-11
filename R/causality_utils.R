library(dagitty)
library(ggdag)
library(adapt4pv)

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

# From: https://github.com/Scriddie/Varsortability/blob/main/src/varsortability.py
# With a little help from chatGPT
sortnregress <- function(X) {
  d <- ncol(X)
  W <- matrix(0, d, d)
  increasing <- order(apply(X, 2, var))
  
  for (k in 2:d) {
    covariates <- increasing[1:(k-1)]
    target <- increasing[k]
    
    LR <- lm(X[, target] ~ X[, covariates])
    weight <- abs(coef(LR)[-1])
    
    x=as.matrix(X[, covariates]) * weight
    if (k == 2){#Add zero column to make glm net work (see Stackoverflow)
      x = cbind(x, rep(0,length(x)))
      cv = cv.glmnet(x=x, y=X[,target])
      opt_la = cv$lambda.min
      fit = glmnet(x=x, y=X[,target], lambda = opt_la)
      #LL <- lars(as.matrix(X[, covariates]) * weight, X[, target], type = "lasso")
      W[covariates, target] <- coef(fit)[2]  * weight
    } else{
      cv = cv.glmnet(x=x, y=X[,target])
      opt_la = cv$lambda.min
      fit = glmnet(x=x, y=X[,target], lambda = opt_la)
      #LL <- lars(as.matrix(X[, covariates]) * weight, X[, target], type = "lasso")
      W[covariates, target] <- coef(fit)[-1] * weight
    }
  }
  
  return(W)
}



# From: https://github.com/Scriddie/Varsortability/blob/main/src/varsortability.py
# With a little help from chatGPT
varsortability <- function(X, W, tol=1e-9) {
  # Takes n x d data and a d x d adjacency matrix,
  # where the i,j-th entry corresponds to the edge weight for i->j,
  # and returns a value indicating how well the variance order
  # reflects the causal order.
  E <- W != 0
  Ek <- E
  var <- apply(X, 2, var)
  n_paths <- 0
  n_correctly_ordered_paths <- 0
  
  #The original python code was:
  #var / var.T
  M = kronecker(t(var), var, FUN = "/") #An Entry > 1 is a pair where the "to" has higher variance then the "from"
  #Loops over possible directed path of length 1,2,...|E|-1 |E| number of nodes
  for (i in 1:(nrow(E) - 1)) { 
    n_paths <- n_paths + sum(Ek)
    n_correctly_ordered_paths <- n_correctly_ordered_paths + sum(Ek & M > 1 + tol)
    n_correctly_ordered_paths <- n_correctly_ordered_paths + 
      0.5 * sum(Ek & (M <= 1 + tol) & (M > 1 - tol)) #Borderline 
    Ek <- Ek %*% E #Considering pathes of an increased length
  }
  return(n_correctly_ordered_paths / n_paths)
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


