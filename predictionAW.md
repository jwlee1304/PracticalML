---
title: "Prediction Assignment Writeup"
output:
  html_document:
    keep_md: yes
  pdf_document: default
---

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv



## Data Processing

*Load Libraries*


```r
library(caret)
library(rattle)
```

*Load Data*

```r
# Load train and test data
train <- read.table("pml-training.csv", sep = ",", head = TRUE, na.strings = c("NA", ""))
test <- read.table("pml-testing.csv", sep = ",", head = TRUE, na.strings = c("NA", ""))
```

*Remove missing records*


```r
train <- train[, (colSums(is.na(train)) == 0)]
test <- test[, (colSums(is.na(test)) == 0)]
```


```r
names(train)
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
##  [7] "num_window"           "roll_belt"            "pitch_belt"          
## [10] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
## [13] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [16] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [19] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [22] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [25] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [28] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [31] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [34] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [37] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [40] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [43] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [46] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [49] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [52] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [55] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [58] "magnet_forearm_y"     "magnet_forearm_z"     "classe"
```

*Drop unused columns*


```r
# Drop columns 1 to 7
train <- train[,-1:-7]
test <- test[,-1:-7]
names(train)
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```

```r
dim(train)
```

```
## [1] 19622    53
```

```r
dim(test)
```

```
## [1] 20 53
```

Split training dataset into a training dataset (75% of the observations) and a validation dataset (25% of the observations). This validation dataset will allow us to perform cross validation when choosing our model.


```r
# Here we create a partition of the traning data set 
set.seed(9999)
inTrain  <- createDataPartition(train$classe, p=0.75, list=FALSE)
train1 <- train[inTrain,] #Training Data
test1 <- train[-inTrain,] #Validation Data
dim(train1)
```

```
## [1] 14718    53
```

```r
dim(test1)
```

```
## [1] 4904   53
```


Using 3 different models: 
*1. Classification* 
*2. Random Forest* 
*3. Gradient Boosting* 

We will use cross validation to overcome any overfitting with 5 or 10 folds


```r
trControl <- trainControl(method="cv", number=5)
```

## Model 1: Classification (CL)

```r
model_CL <- train(classe~., data=train1, method="rpart", trControl=trControl)
```


```r
fancyRpartPlot(model_CL$finalModel)
```

![](predictionAW_files/figure-html/unnamed-chunk-9-1.png)<!-- -->


```r
train_class_pred <- predict(model_CL,newdata=test1)
conf_matrix_class <- confusionMatrix(test1$classe,train_class_pred)

# display confusion matrix and model accuracy
conf_matrix_class$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1264   24  104    0    3
##          B  434  303  212    0    0
##          C  397   31  427    0    0
##          D  370  141  293    0    0
##          E  119  125  245    0  412
```


```r
conf_matrix_class$overall[1]
```

```
##  Accuracy 
## 0.4906199
```
The Accuracy with Classification is on 49.06% 


## Model 2: Random Forest (rf)

```r
model_RF <- train(classe~., data=train1, method="rf", trControl=trControl, verbose=FALSE)
print(model_RF)
```

```
## Random Forest 
## 
## 14718 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 11775, 11773, 11773, 11775, 11776 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9915069  0.9892564
##   27    0.9913036  0.9889990
##   52    0.9856641  0.9818623
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```


```r
plot(model_RF,main="Accuracy of RF model")
```

![](predictionAW_files/figure-html/unnamed-chunk-13-1.png)<!-- -->


```r
train_rf_pred <- predict(model_RF,newdata=test1)
conf_matrix_rf <- confusionMatrix(test1$classe,train_rf_pred)

# display confusion matrix and model accuracy
conf_matrix_rf$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    7  941    1    0    0
##          C    0    4  850    1    0
##          D    0    0   13  790    1
##          E    0    0    0    2  899
```


```r
conf_matrix_rf$overall[1]
```

```
##  Accuracy 
## 0.9940865
```


```r
names(model_RF$finalModel)
```

```
##  [1] "call"            "type"            "predicted"      
##  [4] "err.rate"        "confusion"       "votes"          
##  [7] "oob.times"       "classes"         "importance"     
## [10] "importanceSD"    "localImportance" "proximity"      
## [13] "ntree"           "mtry"            "forest"         
## [16] "y"               "test"            "inbag"          
## [19] "xNames"          "problemType"     "tuneValue"      
## [22] "obsLevels"       "param"
```


```r
model_RF$finalModel$classes
```

```
## [1] "A" "B" "C" "D" "E"
```



```r
plot(model_RF$finalModel,main="Model Error for Random Forest Model")
```

![](predictionAW_files/figure-html/unnamed-chunk-18-1.png)<!-- -->



```r
# Compute the variable importance 
MostImpVars <- varImp(model_RF)
MostImpVars
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                      Overall
## roll_belt             100.00
## yaw_belt               82.64
## magnet_dumbbell_z      71.06
## magnet_dumbbell_y      65.98
## pitch_belt             63.68
## pitch_forearm          59.28
## magnet_dumbbell_x      57.94
## roll_forearm           55.32
## accel_dumbbell_y       47.28
## magnet_belt_z          45.38
## accel_belt_z           44.58
## roll_dumbbell          42.66
## magnet_belt_y          39.46
## accel_dumbbell_z       39.16
## roll_arm               35.18
## accel_forearm_x        33.51
## yaw_dumbbell           31.07
## accel_dumbbell_x       30.31
## magnet_arm_y           30.01
## total_accel_dumbbell   29.58
```
The Accuracy with Random Forest is around 99.40% 

Optimal number of predictors - The final value used for the model was mtry = 2.
There is no significal increase of the accuracy with 2 predictors and 27 
But the slope decreases more with more than 27 predictors 
Using more than around 30 trees does not reduce the error significantly.

## Model 3: Gradient Boosting (GBM)


```r
model_GBM <- train(classe~., data=train1, method="gbm", trControl=trControl, verbose=FALSE)
print(model_GBM)
```

```
## Stochastic Gradient Boosting 
## 
## 14718 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 11775, 11775, 11774, 11774, 11774 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7503061  0.6834314
##   1                  100      0.8200853  0.7722094
##   1                  150      0.8537178  0.8148485
##   2                   50      0.8563673  0.8180259
##   2                  100      0.9050833  0.8798709
##   2                  150      0.9329401  0.9151379
##   3                   50      0.8956386  0.8678616
##   3                  100      0.9447620  0.9301024
##   3                  150      0.9620876  0.9520350
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```


```r
plot(model_GBM)
```

![](predictionAW_files/figure-html/unnamed-chunk-21-1.png)<!-- -->


```r
train_gbm_pred <- predict(model_GBM,newdata=test1)
conf_matrix_rf <- confusionMatrix(test1$classe,train_gbm_pred)
conf_matrix_rf$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1373   11    5    5    1
##          B   42  883   24    0    0
##          C    0   28  818    9    0
##          D    1    3   21  771    8
##          E    1    5   13   12  870
```


```r
conf_matrix_rf$overall[1]
```

```
## Accuracy 
##  0.96146
```
The Accuracy with GBM is around 96.14%.

##Conclusion
From the 3 models tested, Random forest model has the hightest accuracy. 
We will Random Forest model to predict the values of classe for the test data set.


```r
final_test_Pred <- predict(model_RF,newdata=test)
final_test_Pred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

