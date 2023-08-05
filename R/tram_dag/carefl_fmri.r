source('R/tram_dag/utils.R')
library(R.utils)
DEBUG = FALSE
DEBUG_NO_EXTRA = FALSE
USE_EXTERNAL_DATA = FALSE 

SUFFIX = 'runLaplace_M10_C0.5_N2500'

DROPBOX = 'C:/Users/sick/dl Dropbox/beate sick/IDP_Projekte/DL_Projekte/shared_Oliver_Beate/Causality_2022/tram_DAG/'
DROPBOX = '~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/Causality_2022/tram_DAG/'
if (rstudioapi::isAvailable()) {
  context <- rstudioapi::getSourceEditorContext()
  this_file <- context$path
  print(this_file)
}

M = 10
len_theta = M + 1
bp = make_bernp(len_theta)

######################################
############# DGP ###############
######################################
library(dplyr)

# Path to the subdirectory
subdirectory <- "data/CAREFL_fMRI/"

# Create an empty data frame to store the final result
train <- data.frame()
intervene_data <- data.frame()
# List all the CSV files with the pattern "train_data_*.csv" in the subdirectory
file_list <- list.files(path = subdirectory, pattern = "train_data_\\d+\\.csv", full.names = TRUE)

# Loop through each file
for (file_name in file_list) {
  # Read the CSV file
  temp_data <- read.csv(file_name, header = FALSE)
  
  # Extract the number from the filename
  file_number <- as.numeric(gsub(".*train_data_(\\d+)\\.csv", "\\1", file_name))
  # Add the file_number as a new column
  temp_data$file_number <- file_number
  
  intervene_file_name <- paste0(subdirectory, "intervene_data_", file_number, ".csv")
  intervene_temp_data <- read.csv(intervene_file_name, header = FALSE)
  # Add the file_number as a new column
  intervene_temp_data$file_number <- file_number
  intervene_data <- bind_rows(intervene_data, intervene_temp_data)
  
  # Append the temporary data to the final data frame
  train <- bind_rows(train, temp_data)
}
colnames(train) =  c('CG', 'HG', 'id')
colnames(intervene_data) = c('CG', 'HG', 'id')


# Now final_data contains all the data with an extra column for the file_number
ids = unique(train$id)
maes = rep(NA, length(ids))
maes_deeptrafo = rep(NA, length(ids))
i = 0
for (id in ids){
  i = i + 1
  #id = ids[1]
  tr = train[train$id == id,]
  inter = intervene_data[intervene_data$id == id,]
  
  fit = lm(HG ~ CG, data=tr)
  mae = median(abs(predict(fit, newdata = inter) - inter$HG))
  maes[i] =  mae
  plot(tr$CG, tr$HG, main = paste0("Patient: ", id, 
                                   " slope=", round(coef(fit)[2],3),
                                   " mae=", round(mae,3)))
  points(inter$CG, inter$HG, col='red')
  abline(fit)  
  
  ######### Using Deep Trafo, a straight forward 
  #m = LmNN(HG ~ 0 + CG, data = tr)
  m = ColrNN(HG ~ 0 + CG, data = tr)
  fit(m, epochs=300, validation_split=0)
  
  d = inter
  d$HG = NULL
  res = predict(m, d, type='cdf')
  hgs = as.numeric(names(res))
  
  error = rep(NA, nrow(inter))
  for (j in 1:length(error)){
    #j = 1
    dd = rep(0, length(res))
    for (d in 1:length(res)){
      dd[d] = res[[d]][j]
    }
    if (sum(dd > 0.5) == 0) {
      error[j] = NA
      cat('Hallo ')
    } else{
      poor_mans_med = hgs[min(which(dd > 0.5))]
      error[j] = abs(inter[j,'HG'] - poor_mans_med)
    }
  }
  cat(sum(is.na(error)))
  maes_deeptrafo[i] = median(error, na.rm = TRUE)
}

mean(maes)#0.603816 (python with Linear not Ridge)
sd(maes) / sqrt(length(ids))

mean(maes_deeptrafo)#0.603816 (python with Linear not Ridge)
sd(maes_deeptrafo) / sqrt(length(maes_deeptrafo))







