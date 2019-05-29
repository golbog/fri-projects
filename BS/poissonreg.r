library(ggplot2)
library(rstan)
library(reshape2)
library(psych) 


y = rpois(1000, 1)
tmp = rnorm(1000,40,15)
tmp[which(tmp < 18)] = 18
x = data.frame(age1=tmp,age2=tmp+runif(1000,-5,5),mat1=rnorm(1000,18,3.8),mat2=rnorm(1000,18,3.8))
m = ncol(x)
n = nrow(x)

for (i in 1:ncol(x)) {
  if (i <= 4) x[,i] <- log(x[,i]+0.1) # log
  x[,i] <- (x[,i] - mean(x[,i])) / sd(x[,i]) # standardize
}

pairs.panels(x)

stan_data <- list(y = y, x = x, m = m, n = n)
samples <- stan(file = "./seminarska/poissonreg.stan",  data = stan_data, iter = 1000, warmup = 200,chains=1)
smp = extract(samples)
print(samples, par = c("beta","alpha"))

traceplot(samples, par="beta",nrow = 2, ncol = 5)

tmp = data.frame(smp$beta)
names(tmp) <- names(x)
tmp = melt(tmp)

ggplot(tmp, aes(x = variable, y = value)) + xlab("") + ylab("Posteriorna koeficientov") +
  geom_boxplot() + coord_flip()

x1 = y 
x2 = colMeans(smp$lambda) 

plot(x1 + rnorm(length(x1),0,0.12), x2, xlim = c(0,8), ylim = c(0,8), 
     xlab = "Dejansko št. otrok (razstreseno)", ylab = "Ocenjeno upanje lambde")
