data {
  int<lower=0> n;
  int<lower=0> m;
  matrix[n,m] x; 
  int<lower=0> y[n];
}

parameters {
  real alpha; // intercept
  vector[m] beta;
}

model {
  for (i in 1:n) {
    y[i] ~ poisson(exp(x[i] * beta + alpha));
  }
}

generated quantities {
  real<lower=0> lambda[n];
  int<lower=0> pred[n];
  for (i in 1:n) {
    lambda[i] <- exp(x[i] * beta + alpha);
    pred[i] <- poisson_rng(lambda[i]);
  }
}