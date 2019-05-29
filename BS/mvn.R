require(rstan)
n_subjects <- 500
n_items <- 3
male <-rnorm(n_subjects, mean = 178, sd = 10)
female <- rnorm(n_subjects, mean = 165, sd = 8)
child <- rnorm(n_subjects, mean = 150, sd = 15)
X <- matrix(c(male, female, child), 
            nrow=n_subjects,ncol=n_items)
data2 <- list(d=n_items, n=n_subjects, y=X)
fit2 <- stan(file = './seminarska/mvn.stan', data = data2, iter = 1000,warmup=500,  chains = 1)
print(fit2, par = c("mu", "Sigma"))
traceplot(fit2)
plot(fit2, par = c("corr"))
traceplot_beta<-As.mcmc.list(fit2,pars="corr")
traceplot(traceplot_beta)
plot(As.mcmc.list(fit2))
detach("package:coda",unload=TRUE)
traceplot(fit2, pars=c("mu","Sigma"),ncol=4)
print(fit2)

require(coda)
post_beta<-As.mcmc.list(fit2)
"summary"(post_beta)
effectiveSize(post_beta)
fit2.se_m
plot(post_beta)
detach("package:coda",unload=TRUE)

smp = data.frame(extract(fit2))
plot(smp)

plot(fit2, par = c("corr"))
plot(fit2, par = c("mu","tmpD"))

plot.dataframe <- data.frame(generated=smp$mu.1, real=male)
plot.male <-data.frame(Moški = factor(rep(c("Pravi","Gen."), each=500)), 
           Visina = c(male,rnorm(length(male),mean = smp$mu.1,sd = smp$tmpD.1)))
plot.female <-data.frame(Ženska = factor(rep(c("Pravi","Gen."), each=500)), 
                       Visina = c(female,rnorm(length(male),mean = smp$mu.2,sd = smp$tmpD.2)))
plot.child <-data.frame(Otrok = factor(rep(c("Pravi","Gen."), each=500)), 
                        Visina = c(child,rnorm(length(male),mean = smp$mu.3,sd = smp$tmpD.3)))
p1 <- ggplot(plot.male, aes(x=Visina, fill=Moški)) +
  geom_histogram(binwidth=4, position="dodge") + ylab("Število")
p2 <- ggplot(plot.female, aes(x=Visina, fill=Ženska)) +
  geom_histogram(binwidth=4, position="dodge") + ylab("Število")
p3 <- ggplot(plot.child, aes(x=Visina, fill=Otrok)) +
  geom_histogram(binwidth=4, position="dodge") + ylab("Število")
plot.male <-data.frame(Moški = factor(rep(c("Pravi","Gen."), each=500)), 
                       Visina = c(male,rnorm(length(male),mean = smp$mu.1,sd = smp$tmpD.1)))
plot.female <-data.frame(Ženska = factor(rep(c("Pravi","Gen."), each=500)), 
                          Visina = c(female,rnorm(length(male),mean = smp$mu.2,sd = smp$tmpD.2)))
plot.child <-data.frame(Otrok = factor(rep(c("Pravi","Gen."), each=500)), 
                        Visina = c(child,rnorm(length(male),mean = smp$mu.3,sd = smp$tmpD.3)))
p4 <- ggplot(plot.male, aes(x=Visina, fill=Moški)) +
  geom_histogram(binwidth=4, position="dodge") + ylab("Število")
p5 <- ggplot(plot.female, aes(x=Visina, fill=Ženska)) +
  geom_histogram(binwidth=4, position="dodge") + ylab("Število")
p6 <- ggplot(plot.child, aes(x=Visina, fill=Otrok)) +
  geom_histogram(binwidth=4, position="dodge") + ylab("Število")
plot.male <-data.frame(Moški = factor(rep(c("Pravi","Gen."), each=500)), 
                       Visina = c(male,rnorm(length(male),mean = smp$mu.1,sd = smp$tmpD.1)))
plot.female <-data.frame(Ženska = factor(rep(c("Pravi","Gen."), each=500)), 
                          Visina = c(female,rnorm(length(male),mean = smp$mu.2,sd = smp$tmpD.2)))
plot.child <-data.frame(Otrok = factor(rep(c("Pravi","Gen."), each=500)), 
                        Visina = c(child,rnorm(length(male),mean = smp$mu.3,sd = smp$tmpD.3)))
p7 <- ggplot(plot.male, aes(x=Visina, fill=Moški)) +
  geom_histogram(binwidth=4, position="dodge") + ylab("Število")
p8 <- ggplot(plot.female, aes(x=Visina, fill=Ženska)) +
  geom_histogram(binwidth=4, position="dodge") + ylab("Število")
p9 <- ggplot(plot.child, aes(x=Visina, fill=Otrok)) +
  geom_histogram(binwidth=4, position="dodge") + ylab("Število")

multiplot(p1,p2,p3,p4,p5,p6,p7,p8,p9,cols=3)




# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}
