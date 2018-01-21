---
title: "Assignment"
author: "Sumaiya Sultana Rika"
date: "1/20/2018"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Problem Description

The given datasets contain data from 157 persons diagnosed with different types of breast cancer, including their DNA test data. The original dataset contains 12181 variables for 157 observations which is too large to do any modeling. The second dataset "BreastCancerAll.reduced.using.cfs.arff" is a reduced version of the original one which is the product of feature selection. This reduced dataset contains 60 variables, 59 of them are DNA test data which are numeric or continuous variables and 1 of them is the class or type of the breast cancer, it is a categorical variable.

The predictor variables' values range from 0 to 1 and the class variable has three levels: 
        HER2:50, 
        HR+:54, 
        TN :53. 

The task is to make a classification model to predict the class of the breast cancer from other DNA test data. 

I divide the dataset randomly into two sets: train_set and test_set using the createDataPartition funciton with p=0.8 which means 80% of the dataset goes to the training set and the rest is kept for prediction purpose. The model will be built on the "train_set" and its predictive ability will be checked on the "test_set"" which is unseen to the model.

```{r library}
library(foreign) # to read .arff files
library(caret)   # for cross-validation
library(randomForest) #for randomForest
#library(mlbench)


# dataset load
dataset <- read.arff("/Users/mainulhasan/Desktop/Assignment/data/BreastCancerAll.reduced.using.cfs.arff")

#head(dataset)
#summary(dataset)
set.seed(123)
trainIndex <- createDataPartition(dataset$class, p = 0.8, list = FALSE)
train_set <- dataset[trainIndex,]
test_set <- dataset[-trainIndex,]
# 59 predictor variables
x <- dataset[,1:59] 
#1 target variable 
y <- test_set [,60]


```
## Random Forest
For predictive modeling random forest is used which generates a lot of trees on the training sample and average their outcomes. In order to grow random forests on the "train_set" and check its validation, I use caret package for cross validation and tuning paramters and randomForest for growing random forest. Cross- validation is used so that we can avoid any biased results. The 10-fold cross-validation procedure is repeated 3 times which gives 30 resamples to aggregate the final result. 

## Paramters of Random Forest

The two important paramters of randomForest() are: mtry and ntree. mtry defines the number of variables randomly sampled as candidates at each split and ntree is number of trees to grow. 

## Parameter Tuning: Grid Search
To tune the parameter mtry, I use grid search technique. In the grid search, a grid of paramters is tried.Each axis of the grid is an algorithm parameter, and points in the grid are specific combinations of parameters. Because we are only tuning one parameter, the grid search is a linear search through a vector of candidate values. I set the grid 1:59, to try all 59 paramters to find out for which value of mtry, accuracy is the highest. 

```{r grid search}
set.seed(123)
control <- trainControl(method="repeatedcv",
                        repeats = 3,
                        number=10, 
                        search="grid")
#set.seed(seed)
grid <- expand.grid(.mtry=c(1:59))
rf_gridsearch <- train(class~., 
                       data=train_set, 
                       method="rf", 
                       metric= "Accuracy", 
                       tuneGrid=grid, 
                       trControl=control)
print(rf_gridsearch)

varImp(rf_gridsearch)
plot(rf_gridsearch)
```
```{r variable importance}
varImpObj <- varImp(rf_gridsearch)
plot(varImpObj, main = "Variable Importance of Top 5", top = 5)
```
```{r Out of sample error}
pred <- predict(rf_gridsearch, test_set)
confusionMatrix(pred,y)
missClass = function(values, prediction) {
    sum(prediction != values)/length(values)
}
errRate = missClass(y, pred)
errRate
```

## Conclusion
The model gives a very low out of sample error rate and high out of sample prediction rate. The genrated model can be used to predict other future values as well. But this model does not handle any missing values. So if future observations contain any missing values that should be handled seperately in advance before feeding them in the model. Though the train_set and test_set are set randomly, but it is possible that for different train_Set and test_set, the model gives different accuracy and out of sampe error.
