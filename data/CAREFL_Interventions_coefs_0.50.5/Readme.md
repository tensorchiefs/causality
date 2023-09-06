# Readme

# Readme

Codebase:
[https://github.com/tensorchiefs/carefl/commit/](https://github.com/tensorchiefs/carefl/commit/0bd239773bdffc1)
Please read it from the output file

``` r
format(Sys.time(), "%a %b %d %X %Y")
```

    [1] "Wed Sep 06 14:21:47 2023"

``` r
list.files()
```

    [1] "int_2500_cl_mlp_5_10.p"                                       
    [2] "Readme.qmd"                                                   
    [3] "Readme.rmarkdown"                                             
    [4] "run_carefl_inter_3ba623050d27f581495becfc5ea0fea378ba987f.txt"

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

data <- load_pickle('int_2500_cl_mlp_5_10.p')
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

## The data$XTraining

``` r
head(data$XTraining)
```

                [,1]       [,2]       [,3]       [,4]
    [1,]  0.07263919  0.3979481 -0.1149445 -0.3881790
    [2,]  0.16268828  0.0665062 -0.2039951 -1.6828541
    [3,] -0.11715991  0.2439604 -1.5144415 -1.0952658
    [4,] -0.09428001  1.0821401  0.2922085 -0.7783907
    [5,]  1.85386781 -0.1876810  1.3081930  1.8724458
    [6,]  0.61924803  0.0420919  1.1469577  0.3033089

``` r
summary_stats(as.data.frame(data$XTraining))
```

      Column                Mean          Std_Dev            Kurtosis
    1     V1 -0.0159889323138897  1.0329149971236 0.00117986949599631
    2     V2  0.0051587198948268 0.98052145376413 0.00167760145060213
    3     V3  0.0522721178426408 6.01159684631991   0.125452312851093
    4     V4   0.498054709558131 1.91179798063343 0.00578220171587255
              Shapiro_W            Shapiro_p               Min              Max
    1 0.962783977230874   7.179496752163e-25 -6.25045038004951  6.4982787171781
    2 0.955491962466005 5.66156269960249e-27  -6.2383564299485 6.74534589240224
    3 0.307391813690583 1.26462402913678e-70 -120.166252814025 152.476435169158
    4 0.895015328276807 2.54202540343256e-38 -7.54272587156243 21.6041634544216

``` r
dim(data$XTraining)
```

    [1] 2500    4

## The Interventional Data

The strength of the intervention is indicated. Note that the
intervention is done on X0

    res = mod.predict_intervention(a, n_samples=20000, iidx=0)

``` r
summary_stats(as.data.frame(data$`X_int_-1.0`))
```

      Column                Mean           Std_Dev             Kurtosis Shapiro_W
    1     V1                  -1                 0                  NaN      <NA>
    2     V2 -0.0389491991695912 0.999587679519907 0.000146320457330809      <NA>
    3     V3   -1.10607263166308  4.43081906379835   0.0118546528133636      <NA>
    4     V4   0.546973052334028  1.39958071471105 7.91463553010262e-05      <NA>
      Shapiro_p               Min              Max
    1      <NA>                -1               -1
    2      <NA> -6.17813301086426 5.83254289627075
    3      <NA> -147.194915771484 100.882568359375
    4      <NA> -6.80443811416626 8.22615623474121

## The Other Data

``` r
# Intervention data
data$x3
```

    [1] 0.004626718

``` r
data$x3e
```

    [1] 0.001480384

``` r
data$x4
```

    [1] 0.001383859

``` r
data$x4e
```

    [1] 0.002528634
