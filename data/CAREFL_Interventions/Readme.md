# Readme

# Readme

Codebase:
[https://github.com/tensorchiefs/carefl/commit/](https://github.com/tensorchiefs/carefl/commit/0bd239773bdffc1)
Please read it from the output file

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

## The data$XTraining

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

      Column                Mean          Std_Dev             Kurtosis Shapiro_W
    1     V1  -0.999999940395355                0                  NaN      <NA>
    2     V2 -0.0417615974722496 1.00154215421068 0.000146320467342507      <NA>
    3     V3   -1.09052715725035 3.61717055411497   0.0106368118161091      <NA>
    4     V4   0.342001513559518 1.40877943378615 7.54773854722843e-05      <NA>
      Shapiro_p                Min                Max
    1      <NA> -0.999999940395355 -0.999999940395355
    2      <NA>  -6.19294881820679   5.84121131896973
    3      <NA>  -112.528465270996   83.0536651611328
    4      <NA>  -6.42077922821045   8.75392818450928

## The Other Data

``` r
# Intervention data
data$x3
```

    [1] 0.005312767

``` r
data$x3e
```

    [1] 0.009106333

``` r
data$x4
```

    [1] 0.001516641

``` r
data$x4e
```

    [1] 0.003107906
