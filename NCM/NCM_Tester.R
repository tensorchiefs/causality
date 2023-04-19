# Test code for ParameterizedSCM, MLP, and NCM

# Create a sample adjacency matrix
adj <- matrix(c(0, 1, 0,
                0, 0, 1,
                0, 0, 0), nrow = 3, byrow = TRUE)

# Test ParameterizedSCM
pscm <- ParameterizedSCM$new(adj)
pscm$print_graph()

# Test MLP
input_size <- 4
output_size <- 1
hidden_sizes <- c(10, 10, 10)
mlp <- MLP$new(input_size, output_size, hidden_sizes)

# Test NCM
ncm <- NCM$new(adj)

# Test compute_marginals method
samples <- 100
doX <- -1
Xi <- -1
marginals <- ncm$compute_marginals(samples, doX, Xi)
cat("Computed marginals:\n")
print(marginals)
