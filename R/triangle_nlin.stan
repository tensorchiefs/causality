data {
  int<lower=0> N;
  vector[N] x1;
  vector[N] x2;
  vector[N] x3;
}

parameters {
  real i2; //-1 
  real ni; //1
  real z2; //3
  real e2;//-2
  
  real i3;  //0
  real s1; //1
  real sq; //0.25
  
  real<lower=0> sigma2;
  real<lower=0> sigma3;
}

model {
  x2 ~ normal(-i2 + z2/(ni+exp(e2 * x1)), sigma2);  
  x3 ~ normal(s1*x1 + sq*(x2^2), sigma3);
}

