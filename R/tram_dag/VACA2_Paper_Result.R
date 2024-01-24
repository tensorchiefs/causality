read_and_convert_logs <- function(file_path) {
  # Load reticulate library
  library(reticulate)
  
  # Import Python libraries within R
  pd <- import("pandas")
  json <- import("json")
  
  # Initialize an empty list to hold the dictionaries
  log_dicts <- list()
  
  # Read the file line-by-line and append each line as a dictionary to the list
  con <- file(file_path, "r")
  while (TRUE) {
    line <- readLines(con, n = 1)
    if (length(line) == 0) {
      break
    }
    log_dicts <- append(log_dicts, list(json$loads(line)))
  }
  close(con)
  
  # Convert the list of dictionaries to a DataFrame using pandas
  df_py <- pd$json_normalize(log_dicts)
  
  # Return the Python DataFrame object directly for debugging
  return(df_py)
}

# Usage example
file_path <- "data/VACA2_triangle_nl/25K/wandb_local/logs.txt"

file_path <- "data/VACA1_triangle_lin/carefl/wandb_local/logs.txt"
file_path <- "data/VACA1_triangle_lin/25K/wandb_local/logs.txt"
file_path <- "data/VACA1_triangle_lin/NSF/wandb_local/logs.txt"
df_py <- read_and_convert_logs(file_path)
plot(df_py$val.loss)
new_df <- df_py %>% drop_na(`test.rmse_ate_x1=25_50`)
new_df

# # Finding unique values of the 'epoch' column
# unique_epochs <- as.numeric(unique(unlist(df_py$epoch)))
# min = min(unique_epochs, na.rm = TRUE)
# max = max(unique_epochs, na.rm = TRUE)




read_and_convert_logs <- function(file_path) {
  # Load reticulate library
  library(reticulate)
  
  # Import Python libraries within R
  pd <- import("pandas")
  json <- import("json")
  
  # Initialize an empty list to hold the dictionaries
  log_dicts <- list()
  
  # Read the file line-by-line and append each line as a dictionary to the list
  con <- file(file_path, "r")
  while (TRUE) {
    line <- readLines(con, n = 1)
    if (length(line) == 0) {
      break
    }
    log_dicts <- append(log_dicts, list(json$loads(line)))
  }
  close(con)
  
  # Convert the list of dictionaries to a DataFrame using pandas
  df_py <- pd$json_normalize(log_dicts)
  
  # Check if df_py is NULL
  if (!is.null(df_py)) {
    # Convert the pandas DataFrame to an R data.frame
    df <- py_to_r(df_py)
    return(df)
  } else {
    stop("The DataFrame could not be created.")
  }
}

# Usage example
file_path <- "data/VACA1_triangle_lin/25K/wandb_local/logs.txt"
df <- read_and_convert_logs(file_path)

# Show the first 10 rows and first 5 columns of the data frame
head(df[, 1:5], 10)


# Install and load jsonlite package if you haven't
library(jsonlite)

# Initialize empty list to hold the data
data_list <- list()

# Read each line from the file and convert from JSON to list
con <- file("data/VACA1_triangle_lin/25K/wandb_local/logs.txt", "r")

# Load required packages
library(jsonlite)
library(dplyr)

# Initialize an empty list to hold the dictionaries
log_list <- list()

# Read the file line-by-line and append each line as a dictionary to the list
#con <- file("logs.txt", "r")
while (TRUE) {
  line <- readLines(con, n = 1)
  if (length(line) == 0) {
    break
  }
  
  # Replace NaN with "null" before parsing
  line <- gsub("NaN", "null", line)
  
  log_list <- append(log_list, list(fromJSON(line)))
}
close(con)

# Convert all 'epoch' fields to character to ensure they are of the same type
log_list <- lapply(log_list, function(x) {
  x$epoch <- as.character(x$epoch)
  return(x)
})

# Convert the list of dictionaries to a data frame
df <- bind_rows(log_list)

# Flatten the nested columns
df <- jsonlite::flatten(df)




# Convert list to data frame
data_df <- do.call(rbind, lapply(data_list_filled, data.frame, stringsAsFactors=FALSE))



















library(readr)

X <- read_csv("~/Downloads/metrics.csv", col_names = TRUE)

X <- read_csv("~/Downloads/VACA1_triangle_LIN_data_0.csv", col_names = FALSE)
qqplot(train$df_orig[,1]$numpy(), X$X1)
hist(X$X1)
abline(0,1)


Xobs <- read_csv("data/VACA1_triangle_lin/vaca1_triangle_lin_XobsModel.csv", col_names = FALSE)
hist(Xobs$X1,100)
qqPlot(Xobs$X1)

dd = train$df_orig[,1]$numpy()
library(car)
hist(dd,100)
abline(0,1)
Xobs <- read_csv("~/Downloads/vaca2_triangle_nl_Xobs.csv", col_names = FALSE)

#### DGP | Observational Data ####
# This tests that we work on the same data
mean(Xobs$X1)
library(car)
qqPlot(Xobs$X1)
qqplot(Xobs$X1, train$df_orig[,1]$numpy())
abline(0,1)

qqplot(Xobs$X2, train$df_orig[,2]$numpy())
abline(0,1)

qqplot(Xobs$X3, train$df_orig[,3]$numpy())
abline(0,1)

##### Interventions
X_inter <- read_csv("~/Downloads/vaca2_triangle_nl_Xinter_x1-0.5.csv", col_names = FALSE)

X_inter$X1
qqPlot(X_inter$X2)
qqPlot(X_inter$X3)
hist(20./(1+exp(-X_inter$X2^2 - 0.5))+rnorm(n = 2500))
hist(X_inter$X3, freq = FALSE)

metrics <- read_csv("~/Downloads/metrics.csv")
plot(metrics$epoch, metrics$loss)
metrics$log_prob_true - metrics$log_prob
