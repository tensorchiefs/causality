# Readme

# Readme

Codebase:
<https://github.com/tensorchiefs/carefl/commit/0bd239773bdffc1>

Command:

    (.venv) (base) oli@olivers-MBP-2 carefl % python main.py -i -n 2500 > "run_carefl_inter_$(git rev-parse HEAD).txt"

Files: The picke file contains serveral data

``` r
library(reticulate)
load_pickle <- function(filepath) {

  
  reticulate::py_run_string(paste0("
import pickle
with open('", filepath, "', 'rb') as file:
    data = pickle.load(file)
  "))
  
  return(py$data)
}

data <- load_pickle('int_2500r_cl_mlp_5_10.p')
summary(data)
```

               Length Class  Mode   
    X_int_-2.5 80000  -none- numeric
    X_int_-2.0 80000  -none- numeric
    X_int_-1.5 80000  -none- numeric
    X_int_-1.0 80000  -none- numeric
    X_int_-0.5 80000  -none- numeric
    X_int_0.0  80000  -none- numeric
    X_int_0.5  80000  -none- numeric
    X_int_1.0  80000  -none- numeric
    X_int_1.5  80000  -none- numeric
    X_int_2.0  80000  -none- numeric
    coeffs         2  -none- numeric
    x3             1  -none- numeric
    x4             1  -none- numeric
    x3e            1  -none- numeric
    x4e            1  -none- numeric
    XTraining  10000  -none- numeric
    loss           1  -none- numeric

``` r
head(data$XTraining)
```

                [,1]       [,2]        [,3]       [,4]
    [1,]  0.07263919  0.3979481 -1.30710859 -1.2543328
    [2,]  0.16268828  0.0665062 -0.08431523  0.2417475
    [3,] -0.11715991  0.2439604 -0.65381683 -0.2729676
    [4,] -0.09428001  1.0821401  0.93194559 -0.9254682
    [5,]  1.85386781 -0.1876810  0.79100733  1.2880533
    [6,]  0.61924803  0.0420919 -0.26249344  1.4338968

``` r
summary_stats(as.data.frame(data$XTraining))
```

      Column                Mean          Std_Dev            Kurtosis
    1     V1 -0.0159889323138897  1.0329149971236 0.00117986949599631
    2     V2  0.0051587198948268 0.98052145376413 0.00167760145060213
    3     V3  0.0324901966044672 4.82124013559566   0.117549688637054
    4     V4   0.325902866557566 1.64127338286142 0.00227135305546914
              Shapiro_W            Shapiro_p               Min              Max
    1 0.962783977230874   7.179496752163e-25 -6.25045038004951  6.4982787171781
    2 0.955491962466005 5.66156269960249e-27  -6.2383564299485 6.74534589240224
    3 0.349617511037226 2.19872666821044e-69 -94.9056110665027 119.763378453045
    4 0.957248452832748 1.71439019287619e-26 -7.44347651690084 15.8559864909954
