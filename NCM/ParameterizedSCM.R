library(R6)

ParameterizedSCM <- R6Class(
  "ParameterizedSCM",
  public = list(
    V = NULL,
    S = NULL,
    U = NULL,
    graph = NULL,
    topology = NULL,
    
    initialize = function(adj) {
      alpha <- c('XYZWABCDEFGHIJKLMNOPQRSTUV')
      self$i2n <- function(x) alpha[x + 1]
      
      self$V <- list()
      self$S <- list()
      self$U <- list()
      self$graph <- list()
      
      for (V in seq_len(ncol(adj)) - 1) {
        pa_V <- which(adj[, V] == 1) - 1
        self$graph[[V]] <- pa_V
        
        V_name <- self$i2n(V)
        self$V <- append(self$V, V_name)
        
        U_V <- function(bs) {
          array(runif(bs), c(bs, 1))
        }
        
        self$U[[V_name]] <- U_V
      }
      
      self$topologicalSort()
    },
    
    print_graph = function() {
      cat("The NCM models the following graph:\n")
      for (k in seq_along(self$graph)) {
        cat(paste0("[", paste(self$i2n(self$graph[[k]]), collapse = ", "), "] --> ", self$i2n(k - 1), "\n"))
      }
    },
    
    indices_to_names = function(indices) {
      sapply(indices, function(x) self$i2n(x))
    },
    
    topologicalSortUtil = function(v, visited, stack) {
      visited[v + 1] <- TRUE
      
      for (i in self$graph[[v + 1]]) {
        if (!visited[i + 1]) {
          self$topologicalSortUtil(i, visited, stack)
        }
      }
      
      stack$insert(1, v)
    },
    
    topologicalSort = function() {
      visited <- rep(FALSE, length(self$V))
      stack <- list()
      
      for (i in seq_along(self$V) - 1) {
        if (!visited[i + 1]) {
          self$topologicalSortUtil(i, visited, stack)
        }
      }
      
      self$topology <- rev(stack)
    },
    
    compute_marginals = function(samples, doX = -1, Xi = -1, debug = FALSE) {
      pred_marginals <- list()
      N <- length(self$V)
      
      for (ind_d in seq_len(N) - 1) {
        vals <- list()
        
        for (val in 0:1) {
          domains <- replicate(N - 1, c(0, 1), simplify = FALSE)
          domains <- insert_element_at(domains, ind_d + 1, list(val))
          combinations <- expand.grid(domains, stringsAsFactors = FALSE)
          
          p_comb <- lapply(seq_len(nrow(combinations)), function(ind) {
            c <- as.matrix(combinations[ind, , drop = FALSE])
            pC <- self$forward(c, matrix(rep(doX, samples), ncol = 1), Xi, samples, debug)
            pC
          })
          
          vals <- append(vals, list(sum(unlist(p_comb))))
        }
        
        pred_marginals[[ind_d + 1]] <- vals
      }
      
      if (debug) {
        browser()
      }
      
      pred_marginals
    },
    
    params = function() {
      # This method should be implemented in the subclasses.
    },
    
    forward = function(v, dodoX, Xi, samples, debug=False){
      
    }
  )
)
                       