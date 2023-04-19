data {
  int<lower=0> N;
  vector[N] x1;
  vector[N] x2;
  vector[N] x3;
}

parameters {
  real a21; //x2:=N(a21*x1+i2, sigma2)
  real i2;
  real a32;  //x3:=N(a32*x2 + i3, sigma3)
  real i3;
  
  real<lower=0> sigma2;
  real<lower=0> sigma3;
}

model {
  x2 ~ normal(a21*x1 + i2, sigma2); 
  x3 ~ normal(a32*x2 + i3, sigma3);
}

