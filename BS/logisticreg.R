require(LaplacesDemon)
a <- c(rbern(333,0.5),rbern(333,0.45),rbern(334,0.4))
b <- c(rbern(500,0.5),rbern(500,0.35))
c <- c(rbern(800,0.5),rbern(200,0))


data <- data.frame(turn=seq(1,1000), player=factor(factor(rep(c("A","B","C"), each=1000))), result=c(a,b,c))
contrasts(data$player) = contr.treatment(3)
x = model.matrix(~turn + player,  data)

stan_data <- list(n = nrow(x), 
                  m = ncol(x), 
                  y = data$result, 
                  x = x)

fit <- stan(file = './seminarska/logisticreg.stan',  data = stan_data, iter = 1500, chains = 1)
traceplot(fit, par="beta",ncol=2,nrow=2)

print(fit,pars="beta")
require(coda)
post_beta<-As.mcmc.list(fit)
effectiveSize(post_beta,pars="beta") #EFEKTIVE SIZE
plot(post_beta)
detach("package:coda",unload=TRUE)

a1 <- data.frame(x=c(0,333,333,666,666,1000),y=c(0.5,0.5,0.45,0.45,0.4,0.4))
b1 <- data.frame(x=c(0,500,500,1000), y=c(0.5,0.5,0.35,0.35))
c1 <- data.frame(x=c(0,800,800,1000),y=c(0.5,0.5,0,0))

ggplot(a1, aes(x=x,y=y,color="A")) + geom_line() + 
  geom_line(data=b1, aes(x=x,y=y,color="B")) + 
  geom_line(data=c1, aes(x=x,y=y,color="C")) + theme(legend.title=element_blank()) + xlab("Zaporedna številka meta") + ylab("Verjetnost")

smp = extract(fit)
tmp = data.frame(data, Utility = colMeans(smp$eta), Prob = colMeans(smp$theta))
ggplot(tmp, aes(x = turn, y = result)) + 
  geom_point() + facet_wrap(~player) + 
  geom_line(data = tmp, aes(x = turn, y = Utility, colour = "red")) + theme_bw() +
  geom_line(data = tmp, aes(x = turn, y = Prob, colour = "blue")) + theme(legend.position = "none") + xlab("Met") + ylab("Rezultat") + geom_jitter(height = 0.1)
