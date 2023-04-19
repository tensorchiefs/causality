data {
  int<lower=0> N;
  vector[N] x1;
  vector[N] x2;
  vector[N] x3;
}

parameters {
  //x2:=N(a21*x1+i2, sigma2)
  real a21; 
  real i2;
  real<lower=0> sigma2;
  
  //x3:=N(a32*x2 + a31*x1 + i3, sigma3)
  real a32;  
  real a31;
  real i3;
  real<lower=0> sigma3;
  
  //x2:=w*N(m1, sigma11) + (1-w)*N(m2, sigma12)
  real m1;  
  real m2;
  real<lower=0, upper=1> w;
  real<lower=0> sigma11;
  real<lower=0> sigma12;
}

model {
  for (n in 1:N) {
    target += log_mix(w,
                     normal_lpdf(x1[n] | m1, sigma11),
                     normal_lpdf(x1[n] | m2, sigma12));
    target += normal_lpdf(x2[n] | a21*x1+i2, sigma2);
    target += normal_lpdf(x3[n] | a32*x2 + a31*x1 + i3, sigma3);
  }
  //x1 ~ w*normal(m1, sigma11) + (1-w)*normal(m2, sigma12)
  //x2 ~ normal(a21*x1+i2, sigma2); # a21*x1+i2 + u2  i.e. x2 = f_theta2(x1) + u2
  //x3 ~ normal(a32*x2 + a31*x1 + i3, sigma3);
}

