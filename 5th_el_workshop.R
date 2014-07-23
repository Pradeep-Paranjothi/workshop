# Create R Project

version
# Find current working directory
getwd()

#Set the working directory
?setwd
setwd("/tmp/")

#LOAD THE DATA
input_data <- read.csv("5th_el_train.csv")

# time it takes to load
system.time(input_data <- read.csv("5th_el_train.csv"))

head(input_data)

#library(data.table)

# Finding help
?

# data frame, matrix, vector

# set.seed to generate the same random numbers 
set.seed(12345)
train_list <- sample(1:150000,nrow(input_data)*0.8)

unique(train_list)
nrow(data.frame(unique(train_list)))

# Create training dataset
train <- input_data[train_list,]
test <- input_data[-(train_list),]

dim(train)
nrow(train)
ncol(train)

class(train)
class(train[,2])
names(train)
head(train)
head(train[,1:5],10)

for(i in 1:12)
{
	print("------")
	print(colnames(train)[i])
	print(class(train[,i]))
	print(summary(train[,i]))
}


lapply(train,class)


# Data Types: integer, numeric, factor, character

lapply(train,summary)

max(input_data$MonthlyIncome)
max(input_data$MonthlyIncome,na.rm=T)

# Printing the record that has the lowest age
subset(train,train$age==min(train$age))


colnames(train)

# renaming column name
names(train)[10] <- "num_bw_60_90"
names(test)[10] <- "num_bw_60_90"

# Plot/Summarize and see how the distribution looks

#Freq dist
table(train[,12])
table(input_data[,12],input_data[,1])

#Bar Plot
counts <- table(train[,12])
barplot(counts,main="Distribution of X1",xlab="# Number")

# Histogram
hist(train[,3])


plot(train$MonthlyIncome)


plot(train[train$MonthlyIncome<50000,6])

#Capping outliers
train$MonthlyIncome <- replace(train$MonthlyIncome,train$MonthlyIncome>50000,50000)

#Handling outliers in numeric variables
#Cap the highest value at 95th percentile and lowest value at 1st percentile

fun <- function(x){
    quantiles <- quantile( x, c(.01, .99 ),na.rm=TRUE )
    x[ x < quantiles[1] ] <- quantiles[1]
    x[ x > quantiles[2] ] <- quantiles[2]
    x
}
train[,3] <- fun( train[,3] )
test[,3] <- fun( test[,3] )



library(sqldf)
sqldf("select count(*) from train where MonthlyIncome <= 50000")

summary(train$MonthlyIncome)

# missing values
apply(is.na(train[,]),2,sum)

# impute missing values
missing_cols <- c(6,11)
train[,missing_cols] <- apply(data.frame(train[,missing_cols]), 2, function(x){x <- replace(x, is.na(x), mean(x, na.rm=TRUE))}) 

# Check if everything's fine
lapply(train,summary)

# Checking first few rows
head(train)

# Finding unique values of a column
unique(train[,11])

# Rounding Number of Dependents
train[,11] <- round(train[,11],0)

# Convert categorical to numerical (one-hot encoding)

# First check class
lapply(train,class)

# number of unique values in StateCat
unique(train$StateCat)

# I prefer to use class.ind 
library(nnet)

statecat <- data.frame(class.ind(train$StateCat))

# Another approach
x1 <- model.matrix(~train[,12],data=train)

# If there are k categories, k-1 columns are to be appended to the training data set
train <- data.frame(cbind(train[,-12],statecat[,-4]))
########################################################################################
# Test : Do the same data transformation process for test
test[,missing_cols] <- apply(data.frame(test[,missing_cols]), 2, function(x){x <- replace(x, is.na(x), mean(x, na.rm=TRUE))}) 
test$MonthlyIncome <- replace(test$MonthlyIncome,test$MonthlyIncome>50000,50000)
statecat <- data.frame(class.ind(test$StateCat))

# Another approach
x1 <- model.matrix(~test[,12],data=test)
test <- data.frame(cbind(test[,-12],statecat[,-4]))



# Models


# Build Logistic Regression Model
library(glmnet)

#L1 regression
model_1 <- cv.glmnet(as.matrix(train[,-1]),as.matrix(train[,1]),nfolds=10,family="binomial",type.measure="auc",alpha=1,grouped=FALSE)
max(model_1$cvm)

#L2 regression
model_2 <- cv.glmnet(as.matrix(train[,-1]),as.matrix(train[,1]),nfolds=10,family="binomial",type.measure="auc",alpha=0,grouped=FALSE)
max(model_2$cvm)

#Elastic net regression
model_3 <- cv.glmnet(as.matrix(train[,-1]),as.matrix(train[,1]),nfolds=10,family="binomial",type.measure="auc",alpha=0.95,grouped=FALSE)
max(model_3$cvm)


# Execute CV in parallel
require(doMC)

detectCores()

registerDoMC(cores=7)




#L1 regression
system.time(model_1 <- cv.glmnet(as.matrix(train[,-1]),as.matrix(train[,1]),nfolds=10,family="binomial",type.measure="auc",alpha=1,grouped=FALSE,maxit=10000,
						type.logistic="modified.Newton",thresh=1e-03,parallel=TRUE))
max(model_1$cvm))

#L2 regression
system.time(model_2 <- cv.glmnet(as.matrix(train[,-1]),as.matrix(train[,1]),nfolds=10,family="binomial",type.measure="auc",alpha=0,grouped=FALSE,maxit=10000,
						type.logistic="modified.Newton",thresh=1e-03,parallel=TRUE))
max(model_2$cvm))

#Elastic net regression
system.time(model_3 <- cv.glmnet(as.matrix(train[,-1]),as.matrix(train[,1]),nfolds=10,family="binomial",type.measure="auc",alpha=0.95,grouped=FALSE,maxit=10000,
						type.logistic="modified.Newton",thresh=1e-03,parallel=TRUE))
max(model_3$cvm))

# Time taken: About 43 seconds
# Predict

# To get probability, use response.
prod_1 <- predict(model_1,as.matrix(test[,-1]),s=model_1$lambda.min,type="response")

# To get class, use class
prod_2 <- predict(model_1,as.matrix(test[,-1]),s=model_1$lambda.min,type="class")

prod_2 <- data.frame(prod_2)
prod_2[,1] <- as.numeric(as.character(prod_2[,1]))


# Model evaluation
library(ROCR)
pred <- prediction(prod_2,test[,1])
# Note: Both y2 and actual should be numeric. Not factors
RP.perf <- performance(pred,"prec","rec")
plot(RP.perf)

ROC.perf <- performance(pred,"tpr","fpr")
plot(ROC.perf)

AUC.perf <- performance(pred,"auc")
auc <- as.numeric(AUC.perf@y.values)
auc

performance(pred,"f")


precision <- sum(prod_2[,1] & test[,1]) / sum(prod_2[,1])
recall <- sum(prod_2[,1] & test[,1]) / sum(test[,1])


# Decision Tree
library(rpart)
model_9 <- rpart(Default~.,data=train,method="class")

printcp(model_9)
plotcp(model_9)
summary(model_9)

# plot tree 
plot(model_9, uniform=TRUE, 
  	main="Classification Tree ")
text(model_9, use.n=TRUE, all=TRUE, cex=.8)




#RANDOM FOREST
library(randomForest)
system.time(model_3 <- randomForest(train[,-1],as.factor(train[,1]),ntree=50))
plot(model_3)
# Time: 19 seconds
prod_3 <- predict(model_3,test[,-1],type="response")
prod_3 <- data.frame(prod_3)


#knn
library(caret)
model_8 <- knn3(as.matrix(train[,-1]),as.factor(train[,1]))
model_8

y1 <- predict(model_8,test[,-1],type="prob")
y2 <- predict(model_8,test[,-1],type="class")

confusionMatrix(y2,test[,1])

actual <- as.factor(test[,1])

library(ROCR)
pred <- prediction(y2,actual)
# Note: Both y2 and actual should be numeric. Not factors
RP.perf <- performance(pred,"prec","rec")
plot(RP.perf)

ROC.perf <- performance(pred,"tpr","fpr")
plot(ROC.perf)

AUC.perf <- performance(pred,"auc")
auc <- as.numeric(AUC.perf@y.values)
auc

performance(pred,"f")

predict <- prod_2
true <- test[,1]

precision <- sum(predict & true) / sum(predict)
recall <- sum(predict & true) / sum(true)
Fmeasure <- 2 * precision * recall / (precision + recall)



#Support Vector Machines
library(e1071)
system.time(model_4 <- svm(train[,-1],as.factor(train[,1]),type="C-classification",kernel="radial",cost=10,probability=TRUE,tolerance=0.1,shrinking=FALSE))

library(kernlab)



system.time(model_4a <- ksvm(as.matrix(train[,-1]),as.matrix(train[,1]),type="C-svc",C=1,scale=FALSE,kernel="rbfdot",prob.model=TRUE))
system.time(model_4a <- ksvm(as.matrix(train[,-1]),as.matrix(train[,1]),type="C-svc",C=1,
			scale=TRUE,kernel="rbfdot",prob.model=TRUE, kpar=list(sigma=0.05),nu=0.2,epsilon=0.5,cache=200,tol=0.1))
# Time: 199 sec


# Create more features. Do a two-way multiplication

a <- c(1,2,3,4,5)
x <- outer(a,a)
x[lower.tri(x)]
# Function to create all 2-way product features

feature.gen <- function (x){
  x <- outer (x, x)
  x [lower.tri (x)]
}
train_two_way <- as.matrix(cbind(t(apply (train[,-1], 1, feature.gen)),train))


# Find principal component

train_pca <- prcomp(train_two_way,scale=TRUE)

# error: unit variance. Check what columns have unit variance and eliminate them

train_var <- apply(train_two_way,2,var)
train_mean <- apply(train_two_way,2,mean)
train_con <- data.frame(cbind(train_var,train_mean))
train_con$diff <- train_con[,1] - train_con[,2]

# 76,77,78 are unit variance columns

train_1 <- data.frame(train_two_way[,-c(76:78)])

train_pca <- prcomp(train_1,scale=TRUE)

summary(train_pca)
# select 90% variation

train_with_pca <- data.frame(predict(train_pca)[,1:50])

# Fit Model

test_two_way <- as.matrix(cbind(t(apply (test[,-1], 1, feature.gen)),test))
test_1 <- data.frame(test_two_way[,-c(76:78)])
test_with_pca <- data.frame(predict(train_pca,test_1))[,1:50]


# Feature selection (foba)

library(foba)

system.time(foba_train <- foba(train_two_way[,-79],train_two_way[,79]))
# 130 sec
y1 <- data.frame(predict(foba_train,as.matrix(train_two_way[,-79]),k=20,type="fit"))
summary(y1)
output_foba <- data.frame(y1$fit)

# use this in glmnet
c1 <- data.frame(y1$selected.variables)

train_foba_glmnet <- train_two_way[,c1[,1]]

# Now use train_foba_glmnet as input training dataset


# boosting (gbm)
library(gbm)
model_6 <- gbm.fit(train[,-1],train[,1],distribution="bernoulli",n.trees=50,interaction.depth=1,
					shrinkage=0.01,bag.fraction=0.5,train.fraction=0.8,keep.data=FALSE,verbose=TRUE)


# Instead of using gbm.fit, running gbm to find optimal number of trees based on cross validation
model_7 <- gbm(Default~.,data=train,distribution="bernoulli",n.trees=50,interaction.depth=1,cv.folds=7,class.stratify.cv=TRUE,
					shrinkage=0.01,bag.fraction=0.5,train.fraction=1,keep.data=FALSE,verbose=TRUE,n.cores=7)


iterations_optimal <- gbm.perf(object = model_7 ,plot.it = TRUE,oobag.curve = TRUE,overlay = TRUE,method="cv")
print(iterations_optimal)



########################################################################################




