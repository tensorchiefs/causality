##### Oliver's MAC ####
reticulate::use_python("/Users/oli/miniforge3/envs/r-tensorflow/bin/python3.8", required = TRUE)
library(reticulate)
reticulate::py_config()


#### A mixture of discrete and continuous variables ####
library(tensorflow)
library(keras)
library(mlt)
library(tram)
library(MASS)
library(tensorflow)
library(keras)
library(tidyverse)
source('summerof24/utils_tf.R')

#### For TFP
library(tfprobability)
source('summerof24/utils_tfp.R')
##### Flavor of experiment ######

#### Saving the current version of the script into runtime
DIR = 'summerof24/runs/vaca_triangle/run_1'
if (!dir.exists(DIR)) {
  dir.create(DIR, recursive = TRUE)
}
# Copy this file to the directory DIR
file.copy('summerof24/vaca_triangle_s24.R', file.path(DIR, 'vaca_triangle_s24.R'), overwrite=TRUE)

num_epochs <- 10000L
len_theta = 30 # Number of coefficients of the Bernstein polynomials
hidden_features_I = c(2,25,25,2)    #hidden_features_CS=hidden_features_I = c(2,25,25,2)
hidden_features_CS = c(2,25,25,2)

SEED = -1 #If seed > 0 then the seed is set


MA =  matrix(c(
  0, 'ci', 'ci', 
  0,    0, 'ci', 
  0,    0,   0), nrow = 3, ncol = 3, byrow = TRUE)
MODEL_NAME = 'ModelCI'


# fn = 'triangle_mixed_DGPLinear_ModelLinear.h5'
# fn = 'triangle_mixed_DGPSin_ModelCS.h5'

if (SEED < 0){
  fn = file.path(DIR, paste0('vaca_triangle', '_', MODEL_NAME))
} else{
  fn = file.path(DIR, paste0('vaca_triangle', FUN_NAME, '_', MODEL_NAME, '_SEED', SEED))
}
print(paste0("Starting experiment ", fn))
   
##### DGP ########
dgp <- function(n_obs, doX=c(NA, NA, NA), seed=-1) {
    if (seed > 0) {
      set.seed(seed)
      print(paste0("Setting Seed:", seed))
    }
    print("=== Using the DGP of the VACA1 paper in the linear Fashion (Tables 5/6)")
    flip = sample(c(0,1), n_obs, replace = TRUE)
    X_1 = flip*rnorm(n_obs, -2, sqrt(1.5)) + (1-flip) * rnorm(n_obs, 1.5, 1)
    if (is.na(doX[1]) == FALSE){
      X_1 = X_1 * 0 + doX[1]
    }
    X_2 = -X_1 + rnorm(n_obs)
    if (is.na(doX[2]) == FALSE){
      X_2 = X_2 * 0 + doX[2]
    }
    X_3 = X_1 + 0.25 * X_2 + rnorm(n_obs)
  
  
    #hist(X_3)
    A <- matrix(c(
      0,1,1,
      0,0,1,
      0,0,0
    ), nrow = 3, ncol = 3, byrow = TRUE)
    dat.orig =  data.frame(x1 = X_1, x2 = X_2, x3 = X_3)
    dat.tf = tf$constant(as.matrix(dat.orig), dtype = 'float32')
    
    q1 = quantile(dat.orig[,1], probs = c(0.05, 0.95)) 
    q2 = quantile(dat.orig[,2], probs = c(0.05, 0.95))
    q3 = quantile(dat.orig[,3], probs = c(0.05, 0.95))
    
    
    return(list(
      df_orig=dat.tf, 
      df_R = dat.orig,
      #min =  tf$reduce_min(dat.tf, axis=0L),
      #max =  tf$reduce_max(dat.tf, axis=0L),
      min = tf$constant(c(q1[1], q2[1], q3[1]), dtype = 'float32'),
      max = tf$constant(c(q1[2], q2[2], q3[2]), dtype = 'float32'),
      type = c('c', 'c', 'c'),
      A=A))
} 

train = dgp(2500, seed=ifelse(SEED > 0, SEED, -1))
test  = dgp(2500, seed=ifelse(SEED > 0, SEED + 1, -1))
(global_min = train$min)
(global_max = train$max)
data_type = train$type


len_theta_max = len_theta
for (i in 1:nrow(MA)){ #Maximum number of coefficients (BS and Levels - 1 for the ordinal)
  if (train$type[i] == 'o'){
    len_theta_max = max(len_theta_max, nlevels(train$df_R[,i]) - 1)
  }
}
param_model = create_param_model(MA, hidden_features_I = hidden_features_I, 
                                 len_theta = len_theta_max, 
                                 hidden_features_CS = hidden_features_CS)

x = tf$ones(shape = c(2L, 3L))
param_model(1*x)
MA
h_params = param_model(train$df_orig)
# Check the derivatives of h w.r.t. x
x <- tf$ones(shape = c(2L, 3L)) #B,P
with(tf$GradientTape(persistent = TRUE) %as% tape, {
  tape$watch(x)
  y <- param_model(x)
})
# parameter (output) has shape B, P, k (num param)
# derivation of param wrt to input x
# input x has shape B, P
# derivation d has shape B,P,k, B,P
d <- tape$jacobian(y, x)
d[1,,,2,] # only contains zero since independence of batches


# loss before training
struct_dag_loss(t_i=train$df_orig, h_params=h_params)

if (FALSE){
  with(tf$GradientTape(persistent = TRUE) %as% tape, {
    h_params = param_model(train$df_orig)
    loss = struct_dag_loss(train$df_orig, h_params)
  })
  gradients = tape$gradient(loss, param_model$trainable_variables)
  gradients
}

param_model = create_param_model(MA, hidden_features_I=hidden_features_I, len_theta=len_theta, hidden_features_CS=hidden_features_CS)
optimizer = optimizer_adam()
param_model$compile(optimizer, loss=struct_dag_loss)
param_model$evaluate(x = train$df_orig, y=train$df_orig, batch_size = 7L)


##### Training or readin of weights if h5 available ####
fnh5 = paste0(fn, '_E', num_epochs, '.h5')
fnRdata = paste0(fn, '_E', num_epochs, '.RData')
if (file.exists(fnh5)){
  param_model$load_weights(fnh5)
  load(fnRdata) #Loading of the workspace causes trouble e.g. param_model is zero
  # Quick Fix since loading global_min causes problem (no tensors as RDS)
  (global_min = train$min)
  (global_max = train$max)
} else { #Training
  hist = param_model$fit(x = train$df_orig, y=train$df_orig, epochs = num_epochs,verbose = TRUE)
  plot(hist$epoch, hist$history$loss)
  # Save the model
  param_model$save_weights(fnh5)
  train_loss = hist$history$loss
  val_loss = hist$history$val_loss
  save(val_loss, train_loss, MA, len_theta,
       hidden_features_I,
       hidden_features_CS,
       #global_min, global_max,
       file = fnRdata)
}


####### FINISHED TRAINING #####
#pdf(paste0('loss_',fn,'.pdf'))
epochs = length(train_loss)
plot(1:length(train_loss), train_loss, type='l', main='Training (black: train, green: valid)')
plot(1:length(train_loss), train_loss, type='l', main='Training (black: train, green: valid)', xlim = c(9000,10000), ylim=c(1.65,1.72))
#lines(1:length(train_loss), val_loss, type = 'l', col = 'green')


##### Checking observational distribution ####
library(car)
s = do_dag_struct(param_model, train$A, doX=c(NA, NA, NA), num_samples = 5000)
par(mfrow=c(1,3))
for (i in 1:3){
  d = s[,i]$numpy()
  hist(train$df_orig$numpy()[,i], freq=FALSE, 100,main=paste0("X",i, " red: ours, black: data"), xlab='samples')
  lines(density(train$df_orig$numpy()[,i]), col='blue')
  #hist(train$df_orig$numpy()[,i], freq=FALSE, 100,main=paste0("X_",i))
  lines(density(s[,i]$numpy()), col='red')
  #qqplot(train$df_orig$numpy()[,i], s[,i]$numpy())
  #abline(0,1)
}
par(mfrow=c(1,1))

##### Plot for paper ####
Xref <- as.matrix(read_csv("data/VACA1_triangle_lin/25K/VACA1_triangle_LIN_XobsModel.csv", col_names = FALSE))
names <- c("Ours", "CNF", "DGP")  
custom_colors <- c("Ours" = "#1E88E5", "CNF" = "#FFC107", "DGP" = "red")  

XDGP = dgp(25000)$df_orig$numpy()
Xmodel = s$numpy()

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
      if (j == 3) {
        p = p + xlim(-6, 6)
      }
      if (i == 3 & j < 3) {
        p = p + ylim(-6, 6)
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

library(ggpubr)
g = createPlotMatrix(all_data, "Type", c("X1", "X2", "X3"))
g
filename = paste0(fnh5, "_observations.pdf")
ggsave(filename)
if (FALSE){
  file.copy(filename, '~/Dropbox/Apps/Overleaf/tramdag/figures/', overwrite = TRUE)
}

############## Do Effects ########


doX=c(1, NA, NA)
look_at = 3L
s_dag = do_dag_struct(param_model, train$A, doX)
sdgp = dgp(10000, doX=doX, seed=SEED)
median(sdgp$df_orig$numpy()[,look_at])
median(s_dag[,look_at]$numpy())
boxplot(sdgp$df_orig$numpy()[,look_at], notch = TRUE)
abline(h=-0.110043)

boxplot(s_dag[,look_at]$numpy(), notch = TRUE, ylim=c(-5,5))
abline(h=-0.726061)

hist(sdgp$df_orig$numpy()[,look_at], freq=FALSE, 50, xlab='samples', 
     sub='Histogram from DGP with do. red:TRAM_DAG', col='blue')

sample_dag = s_dag[,look_at]$numpy()
lines(density(sample_dag), col='red', lw=2)
lines(density(sdgp$df_orig$numpy()[,look_at]), col='blue')

library(car)
qqplot(sdgp$df_orig$numpy()[,look_at], sample_dag, xlim=c(-5,5), ylim=c(-5,5))
rug(quantile(train$df_orig$numpy()[,look_at], c(0.05, 0.95)), col='red')
abline(0,1)


####### Figure 2 (\label{fig:VACA1Triangle_L2}) #####################
dox_origs = c(-3,-1, 0)
num_samples = 25142L

#### Sampling for model and DGP
inter_mean_dgp_x2 = inter_mean_dgp_x3 = inter_mean_ours_x2 = inter_mean_ours_x3 = NA*dox_origs
inter_dgp_x2 = inter_dgp_x3 = inter_ours_x2 = inter_ours_x3 = matrix(NA, nrow=length(dox_origs), ncol=num_samples)

for (i in 1:length(dox_origs)){
  # i = 1
  ### Our Model
  dox_orig = dox_origs[i]
  s = do_dag_struct(param_model, MA, doX=c(NA, dox_orig, NA), num_samples = num_samples)
  inter_ours_x2[i,] = s$numpy()[,2]
  inter_ours_x3[i,] = s$numpy()[,3]
  
  ### DGP
  #d = dgp(num_samples,doX2=dox_orig)
  res = dgp(num_samples, doX = c(NA, dox_orig, NA))$df_orig$numpy()
  inter_dgp_x2[i,] = res[,2]
  inter_dgp_x3[i,] = res[,3]
}

summary(inter_dgp_x3[1,])
summary(inter_ours_x3[1,])
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
  ok_index = inter_ours_x3[step,, drop=TRUE] > -10 & inter_ours_x3[step,, drop=TRUE] < 10
  df_do = rbind(df_do, data.frame(
    dox = dox_origs[step],
    x2 = inter_ours_x2[step,ok_index],
    x3 = inter_ours_x3[step,ok_index], # #inter_ours_x3[step,],
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




HIER ALTES ZEUGS
######### Simulation of do-interventions #####
doX=c(0.2, NA, NA)
dx0.2 = dgp(10000, doX=doX, seed=SEED)
dx0.2$df_orig$numpy()[1:5,]


doX=c(0.7, NA, NA)
dx7 = dgp(10000, doX=doX, seed=SEED)
#hist(dx0.2$df_orig$numpy()[,2], freq=FALSE,100)
mean(dx7$df_orig$numpy()[,2]) - mean(dx0.2$df_orig$numpy()[,2])  
mean(dx7$df_orig$numpy()[,3]) - mean(dx0.2$df_orig$numpy()[,3])  

########### Do(x1) seems to work#####

#### Check intervention distribution after do(X1=0.2)
df = data.frame(train$df_orig$numpy())
fit.x2 = Colr(X2~X1,df)
x2_dense = predict(fit.x2, newdata = data.frame(X1 = 0.2), type = 'density')
x2s = as.numeric(rownames(x2_dense))

## samples from x2 under do(x1=0.2) via simulate
ddd = as.numeric(unlist(simulate(fit.x2, newdata = data.frame(X1 = 0.2), nsim = 1000)))
s2_colr = rep(NA, length(ddd))
for (i in 1:length(ddd)){
  s2_colr[i] = as.numeric(ddd[[i]]) #<--TODO somethimes 
}

if(sum(is.na(s2_colr)) > 0){
  stop("Pechgehabt mit Colr, viel Glück und nochmals!")
}

hist(dx0.2$df_orig$numpy()[,2], freq=FALSE, 100, main='Do(X1=0.2) X2',  
     sub='Histogram from DGP with do. Blue: Colr', xlab='samples')
lines(x2s, x2_dense, type = 'l', col='blue', lw=2)

# fit.x3 = Colr(X3 ~ X1 + X2,df)
# newdata = data.frame(
#     X1 = rep(0.2, length(s2_colr)), 
#     X2 = s2_colr)
# 
# s3_colr = rep(NA, nrow(newdata))
# for (i in 1:nrow(newdata)){
#   # i = 2
#   s3_colr[i] = simulate(fit.x3, newdata = newdata[i,], nsim = 1)
# }

s_dag = do_dag_struct(param_model, train$A, doX=c(0.2, NA, NA))
hist(dx0.2$df_orig$numpy()[,2], freq=FALSE, 50, main='X2 | Do(X1=0.2)', xlab='samples', 
     sub='Histogram from DGP with do. red:TRAM_DAG')
sample_dag_0.2 = s_dag[,2]$numpy()
lines(density(sample_dag_0.2), col='red', lw=2)
m_x2_do_x10.2 = median(sample_dag_0.2)


s_dag = do_dag_struct(param_model, train$A, doX=c(0.2, NA, NA))
hist(dx0.2$df_orig$numpy()[,3], freq=FALSE, 50, main='X3 | Do(X1=0.2)', xlab='samples', 
     sub='Histogram from DGP with do. red:TRAM_DAG')
sample_dag_0.2 = s_dag[,3]$numpy()
lines(density(sample_dag_0.2), col='red', lw=2)


###### Comparison of estimated f(x2) vs TRUE f(x2) #######
shift_12 = shift_23 = shift1 = cs_23 = xs = seq(-1,1,length.out=41)
idx0 = which(xs == 0) #Index of 0 xs needs to be odd
for (i in 1:length(xs)){
  #i = 1
  x = xs[i]
  # Varying x1
  X = tf$constant(c(x, 0.5, 3), shape=c(1L,3L)) 
  shift1[i] =   param_model(X)[1,3,2]$numpy() #2=LS Term X1->X3
  shift_12[i] = param_model(X)[1,2,2]$numpy() #2=LS Term X1->X2
  
  #Varying x2
  X = tf$constant(c(0.5, x, 3), shape=c(1L,3L)) 
  cs_23[i] = param_model(X)[1,3,1]$numpy() #1=CS Term
  shift_23[i] = param_model(X)[1,3,2]$numpy() #2-LS Term X2-->X3 (Beate Notation)
}

if (FALSE){
  if (MA[2,3] == 'cs' && F32 == 1){
    # Assuming xs, cs_23, and idx0 are predefined vectors
    # Create a data frame for the ggplot
    df <- data.frame(x2 = xs, cs_23 = cs_23)
    
    # Create the ggplot
    p <- ggplot(df, aes(x = x2, y = cs_23)) +
      geom_line(aes(color = "Complex Shift Estimate"), size = 1) +  
      geom_point(aes(color = "Complex Shift Estimate"), size = 1) + 
      geom_abline(aes(color = "f"), intercept = cs_23[idx0], slope = 0.3, size = 1) +  # Black solid line for 'DGP'
      scale_color_manual(
        values = c("Complex Shift Estimate" = "blue", "f" = "black"),  # Set colors
        labels = c("Complex Shift Estimate", "f(x)")  # Custom legend labels with expression for f(X_2)
      ) +
      labs(
        x = expression(x[2]),  # Subscript for x_2
        y = "~f(x)",  # Optionally leave y-axis label blank
        color = NULL  # Removes the color legend title
      ) +
      theme_minimal() +
      theme(legend.position = "none")  # Correct way to remove the legend
    
    # Display the plot
    p
  } else if (MA[2,3] == 'cs' && F32 != 1){
    # Assuming xs, shift_23, and idx0 are predefined vectors
    # Create a data frame for the ggplot
    df <- data.frame(x2 = xs, 
                     shift_23 = cs_23 + ( -cs_23[idx0] - f(0)),
                     f = -f(xs)
                     )
    # Create the ggplot
    p <- ggplot(df, aes(x = x2, y = shift_23)) +
      #geom_line(aes(color = "Shift Estimate"), size = 1) +  # Blue line for 'Shift Estimate'
      geom_point(aes(color = "Shift Estimate"), size = 1) +  # Blue points for 'Shift Estimate'
      geom_line(aes(color = "f", y = f), ) +  # Black solid line for 'DGP'
      scale_color_manual(
        values = c("Shift Estimate" = "blue", "f" = "black"),  # Set colors
        labels = c("Shift Estimate", "f(x)")  # Custom legend labels with expression for f(X_2)
      ) +
      labs(
        x = expression(x[2]),  # Subscript for x_2
        y = "~f(x)",  # Optionally leave y-axis label blank
        color = NULL  # Removes the color legend title
      ) +
      theme_minimal() +
      theme(legend.position = "none")  # Correct way to remove the legend
    
    # Display the plot
    p
  } else{
    print(paste0("Unknown Model ", MA[2,3]))
  }
 
  
  file_name <- paste0(fn, "_f23_est.pdf")
  # Save the plot
  ggsave(file_name, plot = p, width = 8, height = 8)
  file_path <- file.path("/Users/oli/Library/CloudStorage/Dropbox/Apps/Overleaf/tramdag/figures", basename(file_name))
  ggsave(file_path, plot = p, width = 8/3, height = 8/3)
}
    

par(mfrow=c(2,2))
plot(xs, shift_12, main='LS-Term (black DGP, red Ours)', 
     sub = 'Effect of x1 on x2',
     xlab='x1', col='red')
abline(0, 2)

delta_0 = shift1[idx0] - 0
plot(xs, shift1 - delta_0, main='LS-Term (black DGP, red Ours)', 
     sub = paste0('Effect of x1 on x3, delta_0 ', round(delta_0,2)),
     xlab='x1', col='red')
abline(0, -.2)


if (F32 == 1){ #Linear DGP
  if (MA[2,3] == 'ls'){
    delta_0 = shift_23[idx0] - f(0)
    plot(xs, shift_23 - delta_0, main='LS-Term (black DGP, red Ours)', 
         sub = paste0('Effect of x2 on x3, delta_0 ', round(delta_0,2)),
         xlab='x2', col='red')
    #abline(shift_23[length(shift_23)/2], -0.3)
    abline(0, 0.3)
  } 
  if (MA[2,3] == 'cs'){
    plot(xs, cs_23, main='CS-Term (black DGP, red Ours)', xlab='x2',  
         sub = 'Effect of x2 on x3',col='red')
    
    abline(cs_23[idx0], 0.3)  
  }
} else{ #Non-Linear DGP
  if (MA[2,3] == 'ls'){
    delta_0 = shift_23[idx0] + f(0)
    plot(xs, shift_23 - delta_0, main='LS-Term (black DGP, red Ours)', 
         sub = paste0('Effect of x2 on x3, delta_0 ', round(delta_0,2)),
         xlab='x2', col='red')
    lines(xs, -f(xs))
  } else if (MA[2,3] == 'cs'){
    plot(xs, cs_23 + ( -cs_23[idx0] - f(0) ),
         ylab='CS',
         main='CS-Term (black DGP f2(x), red Ours)', xlab='x2',  
         sub = 'Effect of x2 on x3',col='red')
    lines(xs, -f(xs))
  } else{
    print(paste0("Unknown Model ", MA[2,3]))
  }
}
#plot(xs,f(xs), xlab='x2', main='DGP')
par(mfrow=c(1,1))


if (TRUE){
####### Compplete transformation Function #######
### Copied from structured DAG Loss
t_i = train$df_orig
k_min <- k_constant(global_min)
k_max <- k_constant(global_max)

# from the last dimension of h_params the first entriy is h_cs1
# the second to |X|+1 are the LS
# the 2+|X|+1 to the end is H_I
h_cs <- h_params[,,1, drop = FALSE]
h_ls <- h_params[,,2, drop = FALSE]
#LS
h_LS = tf$squeeze(h_ls, axis=-1L)#tf$einsum('bx,bxx->bx', t_i, beta)
#CS
h_CS = tf$squeeze(h_cs, axis=-1L)

theta_tilde <- h_params[,,3:dim(h_params)[3], drop = FALSE]
theta = to_theta3(theta_tilde)
cont_dims = which(data_type == 'c') #1 2
cont_ord = which(data_type == 'o') #3

### Continiuous dimensions
#### At least one continuous dimension exits
h_I = h_dag_extra(t_i[,cont_dims, drop=FALSE], theta[,cont_dims,1:len_theta,drop=FALSE], k_min[cont_dims], k_max[cont_dims]) 

h = h_I + h_LS[,cont_dims, drop=FALSE] + h_CS[,cont_dims, drop=FALSE]

####### DGP Transformations #######
X_1 = t_i[,1]$numpy()
X_2 = t_i[,2]$numpy()
X_3 = t_i[,3]$numpy()

#h2 = x_2_dash = 5 * x_2 + 2 * X_1
h2_DGP = 5 *X_2 + 2 * X_1
h2_DGP_LS = 2 * X_1
h2_DGP_CS = rep(0, length(X_2))
h2_DGP_I = 5 * X_2

#h(x3|x1,x2) = 0.63*x3 - 0.2*x1 - f(x2)
h3_DGP = 0.63*X_3 - 0.2*X_1 - f(X_2)
h3_DGP_LS = -0.2*X_1
h3_DGP_CS = -f(X_2)
h3_DGP_I = 0.63*X_3


par(mfrow=c(2,2))
plot(h2_DGP, h[,2]$numpy(), main='h2')
abline(0,1,col='red')
confint(lm(h[,2]$numpy() ~ h2_DGP))

#Same for Intercept
plot(h2_DGP_I, h_I[,2]$numpy(), main='h2_I')
abline(0,1,col='red')
confint(lm(h_I[,2]$numpy() ~ h2_DGP_I))

plot(h2_DGP_LS, h_LS[,2]$numpy(), main='h2_LS')
abline(0,1,col='red')
confint(lm(h_LS[,2]$numpy() ~ h2_DGP_LS))

#Same for CS
plot(h2_DGP_CS, h_CS[,2]$numpy(), main='h2_CS')
abline(0,1,col='red')
confint(lm(h_CS[,2]$numpy() ~ h2_DGP_CS))

par(mfrow=c(1,1))


par(mfrow=c(2,2))

plot(h3_DGP, h[,3]$numpy(), main='h3')
abline(0,1,col='red')
confint(lm(h[,3]$numpy() ~ h3_DGP))

plot(h3_DGP_I, h_I[,3]$numpy(), main='h3_I')
abline(0,1,col='red')
confint(lm(h_I[,3]$numpy() ~ h3_DGP_I))

#same for ls  
plot(h3_DGP_LS, h_LS[,3]$numpy(), main='h3_LS')
abline(0,1,col='red')
confint(lm(h_LS[,3]$numpy() ~ h3_DGP_LS))

#same for CS
plot(h3_DGP_CS, h_CS[,3]$numpy(), main='h3_CS')
abline(0,1,col='red')
confint(lm(h_CS[,3]$numpy() ~ h3_DGP_CS))

par(mfrow=c(1,1))

}






