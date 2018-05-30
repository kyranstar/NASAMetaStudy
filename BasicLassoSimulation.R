library(glmnet)
library(readr)
library(dplyr)
library(ggplot2)
currentdata <- read_csv("C:/Users/DrFlats/Downloads/CLASSES/SUMMER/currentdata.csv")

#Specifications
numtrials = 1000
graph.x = seq(from=1,to=numtrials) #For setting up graph

{
#Finding Coefficients of Lasso
Get.Lasso <- function(x.data,y.data) {
  lasso <- glmnet(x.data, y.data, alpha=1)
  hotdamn = coef(lasso, s=.3)
  return(hotdamn)
}

#Finding the remaining variables
FindTrueVar <- function(coefmat) {
  #Removing the intercept as a Predictory Variable
  TrueVars = as.numeric(which(coefmat[-1]!=0))
  return(TrueVars)
  #return(nrow(coefmat))
}

#Finding the number of true variables between the two
Num.Share <- function(vec1,vec2){
  return(( length(union(vec1,vec2)) - length(intersect(vec1,vec2))))
}

#Get Original Data Lasso
{
  matr.data = as.matrix(currentdata)
  x <- select(currentdata,-y)
  xdata <- as.matrix(x)
  ydata <- as.matrix(currentdata$y)
  
  truth = FindTrueVar(Get.Lasso(xdata,ydata))
}

#Preparing simulations generally
{
  col.means = c()
  for(i in (1:ncol(matr.data))){
    #Making vector for means of all columns
    col.means <- c(col.means,mean(matr.data[,i]))
  }
  
cov.matrix = cov(currentdata)
}

#Checks the length of the symmetric difference between two sets of predictor variables
Num.Share <- function(vec1,vec2){
  return(( length(union(vec1,vec2)) - length(intersect(vec1,vec2))))
}

#Setting up for later graph
{
same.count = 0
diff.count = 0
sym1.count = 0
same = c()
diff = c()
sym1 = c()

for(i in (1:numtrials)){
  simul1 = MASS::mvrnorm(n=150,mu=col.means,Sigma=cov.matrix,empirical=FALSE)
  
  new.ydata = simul1[,1]
  new.xdata = simul1[,-1]
  #tester = FindTrueVar(Get.Lasso(xdata,ydata))
  #Just Sad
  tester = FindTrueVar(Get.Lasso(new.xdata,new.ydata))
  
  #Finding bare counts
  if (identical(sort(truth),sort(tester))){
  same.count = same.count + 1
  } else if (Num.Share(sort(truth),sort(tester)) == 1){
  sym1.count = sym1.count + 1
  } else {
  diff.count = diff.count + 1
  }
  
  #Finding proportions for each trials
  same <- c(same,same.count/i)
  sym1 <- c(sym1,sym1.count/i)
  diff <- c(diff,diff.count/i)
}

#Graph that stuff
conjoined = data.frame(prop.same = same,
                       prop.diff = diff,
                       prop.sym1 = sym1,
                       trial = graph.x)
ggplot(conjoined, aes(x=trial)) +
  geom_line(aes(y=prop.same,colour="No Differences")) +
  geom_line(aes(y=prop.sym1,colour="1 Difference")) + 
  geom_line(aes(y=prop.diff,colour="2+ Differences")) +
  geom_line(aes(y=prop.same+prop.sym1,colour="1 or Less Differences")) +
  scale_colour_manual(values=c("",
                               "No Differences"="green","1 Difference"="yellow",
                               "2+ Differences"="red",
                               "1 or Less Differences"="blue")) +
  xlab("Number of Trials")+
  ylab("Percentage") +
  labs(title="Man, this took a while")
}
}

#Precentage of trials one or less symmetric differences
same[length(same)] + sym1[length(sym1)]

#To-Do: Find some concise way to convey variables added/deleted