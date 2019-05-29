data {
	int<lower=0> n; // number of samples
	int<lower=0> m; // number of ind. variables
	int<lower=0,upper=1> y[n]; // made/missed
	matrix[n,m] x; // ind. variables
}
parameters {
	vector[m] beta; // coefficients
}
model {
	for (i in 1:n)
		y[i] ~ bernoulli_logit(x[i]*beta);
}
generated quantities {
	real theta[n];
	real eta[n];
	for (i in 1:n) {
		eta[i] = x[i]*beta;
		theta[i] = inv_logit(eta[i]);
	}
}
