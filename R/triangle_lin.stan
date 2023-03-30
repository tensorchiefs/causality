data {
  int<lower=0> N;
  vector[N] x1;
  vector[N] x2;
  vector[N] x3;
}

parameters {
  real a21; //x2:=N(a21*x1+i2, sigma2)
  real i2;
  real a32;  //x3:=N(a32*x2 + a31*x1 + i3, sigma3)
  real a31;
  real i3;
  
  real<lower=0> sigma2;
  real<lower=0> sigma3;
}

model {
  x2 ~ normal(a21*x1+i2, sigma2); # a21*x1+i2 + u2  i.e. x2 = f_theta2(x1) + u2
  x3 ~ normal(a32*x2 + a31*x1 + i3, sigma3);
}

