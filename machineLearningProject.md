Practical Machine Learning Project - Peer Assessment
========================================================

## Overview
The goal of the project is  to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants and predict the correct /incorrect way of performing barbell lifts.
Tha algorithm for prediction has been created basing on randomForest function from the 'caret' package. The results allow us to predict the performance basing on selected 53 variables.

## Getting and cleaning data

The first step was to read the data and libraries needed for performing the prediction:

```r

library(caret)
library(randomForest)

```

```
## Loading required package: lattice
## Loading required package: ggplot2
## Loading required package: methods
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r

setwd("/Users/barbara/Data")
trainingData<-read.csv("pml-training.csv")
dim(trainingData)
testData<-read.csv("pml-testing.csv")
```
```
## [1] 19622   160
```
```r
testData<-read.csv("pml-testing.csv")
dim(testData)
```
```
## [1]  20 160
```

Cleaning data included changing factor into numeric values, skipping irrelevant data and data with high amount of NA. The result is a data frame with 19622 rows and 54 columns (including 'classe' column). Since there are zero and negative values in the data, logarithms cannot be calculated.


```r
trainingData2<-trainingData[,colSums(is.na(trainingData)) < 1000]

trainingData2<-trainingData2[,c(3, 4, 7:11, 21:42, 49:51, 61:72, 84:93)]
```


Finally the dataset was split into training and validating ('testing') set.

```r
inTrain<-createDataPartition(trainingData2[,1], p=0.75, list=FALSE)
training<-trainingData2[ inTrain,]
testing<-trainingData2[-inTrain,]
```


## Exploratory analysis

Plotting the one-by-one relations showed that it's impossible to find a few major coefficients influencing the result.
The correlation analysis showed that many of the coefficients are correlated (ie. gyros_arm_y, gyros_arm_x; magnet_arm_x, accel_arm_x; magnet_arm_z, magnet_arm_y). 
This indicates that data need multifactor analysis algorithm, such as random forest, or, possibly, processing.



```r
plot(trainingData2[,1]~., trainingData2)

corr<-abs(cor(trainingData2[,-54]))
diag(corr)<-0
which(corr>0.8, arr.ind=TRUE)
```

##Training and validation 

Two random forest analyses, performed both with and without PCA preprocessing, revealed  very similar results. Below I present un-preprocessed data, which did a little better in accuracy and Kappa testing. 


```r
fitForest<-train(training$classe~., data=training, method="rf", prox=TRUE)
```

```r
fitForest

```

```
## Random Forest 

14718 samples
   53 predictor
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Bootstrapped (25 reps) 

Summary of sample sizes: 14718, 14718, 14718, 14718, 14718, 14718, ... 

Resampling results across tuning parameters:

  mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
   2    0.995     0.993  0.001520     0.001925
  27    0.998     0.997  0.000737     0.000934
  53    0.995     0.994  0.002409     0.003053

Accuracy was used to select the optimal model using  the largest value.
The final value used for the model was mtry = 27.
```

```r
fitForest$finalModel
```

```
##  Call:
 randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE) 
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 27

        OOB estimate of  error rate: 0.11%
Confusion matrix:
     A    B    C    D    E  class.error
A 4230    0    0    0    0 0.0000000000
B    4 2832    2    1    0 0.0024656569
C    0    3 2591    1    0 0.0015414258
D    0    0    3 2354    0 0.0012728044
E    0    0    0    2 2695 0.0007415647
```
Validation has been performed on the pprepared 'testing' dataset. The result of prediction, compared with the 'true' data, shows very high accuracy, sensivity and specificity of the model.


```r
pred<-predict(fitForest, testing)
confusionMatrix(pred, testing[,1])
```
## Testing 

To test the model on a test dataset, the data had to be prepared (the number of columns reduced in accordance with the training dataset). The final data frame consists of 53 columns (all the columns from training dataset except for 'classe' data).

```r
testData2<-testData[intersect(colnames(trainingData2),colnames(testData))]
dim(testData2)
```

The following code served for generating the predictions for the automatic submission.
```r
predictions <- predict(fitForest,testData2)
```