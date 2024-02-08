library(keras)
library(tensorflow)
source('R/tram_dag/utils_dag_maf.R')

dgp <- function(n_obs, doX1=NA, doX2=NA) {
  print("=== Using the DGP of the VACA1 paper in the linear Fashion (Tables 5/6)")
  flip = sample(c(0,1), n_obs, replace = TRUE)
  X_1 = flip*rnorm(n_obs, -2, sqrt(1.5)) + (1-flip) * rnorm(n_obs, 1.5, 1)
  if (is.na(doX1) == FALSE){
    X_1 = X_1 * 0 + doX1
  }
  X_2 = -X_1 + rnorm(n_obs)
  if (is.na(doX2) == FALSE){
    X_2 = X_2 * 0 + doX2
  }
  X_3 = X_1 + 0.25 * X_2 + rnorm(n_obs)
  dat.s =  data.frame(x1 = X_1, x2 = X_2, x3 = X_3)
  dat.tf = tf$constant(as.matrix(dat.s), dtype = 'float32')
  scaled = scale_df(dat.tf) * 0.99 + 0.005
  
  A <- matrix(c(0, 1, 1, 0,0,1,0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
  return(list(df_orig=dat.tf,  df_scaled = scaled, A=A))
} 
train = dgp(50000)


library(igraph)
graph <- graph_from_adjacency_matrix(train$A, mode = "directed", diag = FALSE)
plot(graph, vertex.color = "lightblue", vertex.size = 30, edge.arrow.size = 0.5)

#source('tram_scm/model_utils.R')
M = 30L
adjacency <- t(train$A) #We need 0 in upper triangular
param_model = create_theta_tilde_maf(adjacency = adjacency, len_theta = M+1, layer_sizes = layer_sizes)
param_model(train$df_scaled)
optimizer = optimizer_adam()
param_model$compile(optimizer, loss=dag_loss)
param_model$evaluate(x = train$df_scaled, y=train$df_scaled, batch_size = 32L)

##### Training ####
hist = param_model$fit(x = train$df_scaled, y=train$df_scaled, epochs = 500L,verbose = TRUE)
plot(hist$epoch, hist$history$loss)
if (FALSE){
  param_model$save('triangle_test.keras') #Needs saving
  param_model =  keras$models$load_model('triangle_test.keras')
} 


library(tidyverse)
library(ggpubr)
library(gridExtra)
library(grid)
source('R/tram_dag/utils.R')

#### Parameters influencing the prediction ####################
DoX = 1 #The variable on which the do-intervention should occur
DoX = 2

######### Creating the plots #######
######## L1 Observed Plotting the Observed Data and Fits #####
s = do_dag(param_model, train$A, doX=c(NA, NA, NA), num_samples = 1000)$numpy()
s <- s[ rowSums(s >= 0 & s <= 1) == ncol(s), ]

Xmodel = unscale(train$df_orig, s)$numpy()#sampleObs(thetaNN_l, A=train$A, 25000))$numpy()
dim(Xmodel)

XDGP = dgp(1000)$df_orig$numpy() #TODO possible sacling?
Xref <- as.matrix(read_csv("data/VACA1_triangle_lin/25K/VACA1_triangle_LIN_XobsModel.csv", col_names = FALSE))
names <- c("Ours", "CNF", "DGP")  
custom_colors <- c("Ours" = "#1E88E5", "CNF" = "#FFC107", "DGP" = "red")  

Xmodel_df <- as.data.frame(Xmodel)
colnames(Xmodel_df) <- c("X1", "X2", "X3")
Xmodel_df$Type <- names[1]

Xref_df <- as.data.frame(Xref)
colnames(Xref_df) <- c("X1", "X2", "X3")
Xref_df$Type <- names[2]

XDGP_df <- as.data.frame(XDGP)
colnames(XDGP_df) <- c("X1", "X2", "X3")
XDGP_df$Type <- names[3]
all_data <- rbind(Xmodel_df, Xref_df, XDGP_df)

# Function to extract legend
get_legend <- function(my_plot){
  tmp <- ggplot2::ggplot_gtable(ggplot2::ggplot_build(my_plot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}


createPlotMatrix <- function(data, type_col, var_names,text_size = 20, axis_title_size = 18) {
  plot_list <- list()
  for (i in 1:3) {
    for (j in 1:3) {
      if (i == j) {
        p <- ggplot(data, aes_string(x = var_names[i], fill = type_col)) +
          geom_density(alpha = 0.4) +
          scale_fill_manual(values = custom_colors, name = "Methods") +
          theme_minimal() +
          theme(text = element_text(size = text_size), axis.title = element_text(size = axis_title_size)) +
          theme(legend.position = "none")
      } else if (i > j) {
        p <- ggplot(data, aes_string(x = var_names[j], y = var_names[i])) +
          geom_density_2d(aes_string(color = type_col), size = 0.5, breaks = c(0.01, 0.04)) +
          scale_color_manual(values = custom_colors, name = "Methods") +
          theme_minimal() +
          theme(text = element_text(size = text_size), axis.title = element_text(size = axis_title_size)) +
          theme(legend.position = "none")
      } else {
        sub_data <- data[sample(nrow(data), 5000), ]
        p <- ggplot(sub_data, aes_string(x = var_names[j], y = var_names[i], color = type_col)) +
          geom_point(shape = 1, alpha = 0.4) +
          scale_color_manual(values = custom_colors, name = "Methods") +
          theme_minimal() +
          theme(text = element_text(size = text_size), axis.title = element_text(size = axis_title_size)) +
          theme(legend.position = "none")
      }
      plot_list[[paste0(i, "_", j)]] = p
    }
  }
  
  # Combine plots using ggarrange function from ggpubr package
  combined <- ggarrange(
    plotlist = plot_list, 
    ncol = 3, nrow = 3, 
    common.legend = TRUE, 
    legend = "bottom"
  )
  
  return(combined)
}

# Sample function call
#library(cowplot)
g = createPlotMatrix(all_data, "Type", c("X1", "X2", "X3"))
g

####### L2 Do Interventions on X2 #####################
dox_origs = c(-3,-1, 0)
num_samples = 1000L

#### Sampling for model and DGP
inter_mean_dgp_x2 = inter_mean_dgp_x3 = inter_mean_ours_x2 = inter_mean_ours_x3 = NA*dox_origs
inter_dgp_x2 = inter_dgp_x3 = inter_ours_x2 = inter_ours_x3 = matrix(NA, nrow=length(dox_origs), ncol=num_samples)
for (i in 1:length(dox_origs)){
  ### Our Model
  dox_orig = dox_origs[i]
  dox=scale_value(train$df_orig, col=2L, dox_orig) #On X2
  dat_do_x_s = do_dag(param_model, train$A, doX = c(NA,dox,NA), num_samples = num_samples)
  
  df = unscale(train$df_orig, dat_do_x_s)
  inter_ours_x2[i,] = df$numpy()[,2]
  inter_ours_x3[i,] = df$numpy()[,3]

  ### DGP
  d = dgp(num_samples, doX2=dox_orig)
  inter_dgp_x2[i,] = d$df_orig[,2]$numpy()
  inter_dgp_x3[i,] = d$df_orig[,3]$numpy()
}

#### Reformating for ggplot
#Preparing a df for ggplot for selected do-values
df_do = data.frame(dox=numeric(0),x2=numeric(0),x3=numeric(0), type=character(0))
for (step in 1:length(dox_origs)){
  df_do = rbind(df_do, data.frame(
    dox = dox_origs[step],
    x2 = inter_dgp_x2[step,],
    x3 = inter_dgp_x3[step,],
    type = 'DGP'
  ))
  df_do = rbind(df_do, data.frame(
    dox = dox_origs[step],
    x2 = inter_ours_x2[step,],
    x3 = inter_ours_x3[step,],
    type = 'Ours'
  )
  )
}

### Loading the data from VACA2
NSF = TRUE
NSF = FALSE
if (NSF){
  X_inter <- read_csv("data/VACA1_triangle_lin/NSF/vaca1_triangle_lin_Xinter_x2=-3.csv", col_names = FALSE)
} else{
  X_inter <- read_csv("data/VACA1_triangle_lin/25K/vaca1_triangle_lin_Xinter_x2=-3.csv", col_names = FALSE)
}
df_do = rbind(df_do, data.frame(
  dox = -3,
  x2 = X_inter$X2,
  x3 = X_inter$X3,
  type = 'CNF' 
))

if (NSF){
  X_inter <- read_csv("data/VACA1_triangle_lin/NSF/vaca1_triangle_lin_Xinter_x2=-1.csv", col_names = FALSE)
} else{
  X_inter <- read_csv("data/VACA1_triangle_lin/25K/vaca1_triangle_lin_Xinter_x2=-1.csv", col_names = FALSE)
}
df_do = rbind(df_do, data.frame(
  dox = -1,
  x2 = X_inter$X2,
  x3 = X_inter$X3,
  type = 'CNF' 
))

if (NSF){
  X_inter <- read_csv("data/VACA1_triangle_lin/NSF/vaca1_triangle_lin_Xinter_x2=0.csv", col_names = FALSE)
} else{
  X_inter <- read_csv("data/VACA1_triangle_lin/25K/vaca1_triangle_lin_Xinter_x2=0.csv", col_names = FALSE)
}

df_do = rbind(df_do, data.frame(
  dox = 0,
  x2 = X_inter$X2,
  x3 = X_inter$X3,
  type = 'CNF' 
))


text_size = 20
axis_title_size = 18
#geom_density(alpha = 0.4) +
#  scale_fill_manual(values = custom_colors, name = "Methods")

# Custom labeller function
custom_labeller <- function(variable, value) {
  return(paste("doX2 =", value))
}

# Your ggplot code
ggplot(df_do) + 
  geom_density(aes(x=x3, fill=type), alpha=0.4, adjust = 1.5) + 
  xlim(-7, 5) +
  ylab("p(x3|do(x2)") +
  scale_fill_manual(values = custom_colors, name = "Methods") +
  facet_grid(~dox, labeller = custom_labeller) +  # Apply custom labeller here
  facet_grid(~dox, labeller = custom_labeller) +
  theme_minimal() +
  theme(text = element_text(size = text_size),
        axis.title = element_text(size = axis_title_size)) +
  theme(axis.text.x = element_text(angle = 90))#, panel.spacing = unit(1, "lines"))


ggsave(make_fn("dox2_dist_x3.pdf"), width = 15/1.7, height = 6/1.7)
if (FALSE){
  file.copy(make_fn("dox2_dist_x3.pdf"), '~/Dropbox/Apps/Overleaf/tramdag/figures/', overwrite = TRUE)
}






