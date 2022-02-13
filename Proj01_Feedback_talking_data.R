setwd("C:/.../Proj01")
getwd()

# -------------------------#
# data information:
# -------------------------#
  
# ip: ip address of click.
# app: app id for marketing.
# device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
# os: os version id of user mobile phone
# channel: channel id of mobile ad publisher
# click_time: timestamp of click (UTC)
# attributed_time: if user download the app for after clicking an ad, this is the time of the app download
# is_attributed: the target that is to be predicted, indicating the app was downloaded

#--------------------------#
# needs identified in the training dataset:
# -> split the training datset for memory allocation
# -> training data balancing for the target variable
#--------------------------#

library(bigreadr)
library(utils)
library(dplyr)
library(data.table)
library(ggplot2)
library(fasttime)
library(lubridate)
library(tidyverse)
library(lattice)
library(caret)
library(e1071)
library(randomForest)
library(pROC)
library(fastAdaboost)
library(ROCR)

# Training and test files are too large to be loaded into memory at once
# Therefore it will be necessary to apply a method of splitting these files

# Splitting variable into chunks
numChunks <- 5

# removing the header:
totalLines <- nlines('train.csv') - 1
maxLinesChunks <- round(totalLines / numChunks)

# lines/file: 36.980.778

# after dividing the dataset into parts (chunks), we will determine the name of these parts
chunks <- c('train_part1.csv', 'train_part2.csv', 'train_part3.csv', 'train_part4.csv', 'train_part5.csv')

# loading first chunk
chunk <- fread(chunks[1])

# checking dataset columns
colNames <- colnames(chunk)

# checking a sampling on it:
head(chunk)

#-----------------Exploratory Analysis--------------#

# Checking the balance of the target variable
# We will load the chunks, remove duplicate values, if any, and return the amount of each class in the target variable

targetCount <- sapply(chunks, function(x) {
  chunk <- fread(x, col.names = colNames)
  chunk <- chunk[!duplicated(chunk), ]
  table(chunk$is_attributed)
})

# summarizing the results obtained with the transposition of the rows and columns
# for better visualization of the target variable
chunksSummary <- t(targetCount)

# Renaming column names for better understanding
colnames(chunksSummary) <- c('No', 'Yes')

# Showing the results
prop.table(chunksSummary)
View(chunksSummary)


#--------------------------------------------#
#                    No         Yes
#train_part1.csv 0.1995894 0.0005177438
#train_part2.csv 0.1996269 0.0004779610
#train_part3.csv 0.1991522 0.0005639860
#train_part4.csv 0.1995156 0.0004459230
#train_part5.csv 0.1995958 0.0005144285
#--------------------------------------------#


# The result presented very similar proportions between each chunk of the dataset
# To improve the reading of the results, we will group the values obtained to have an analysis of the entire dataset
# Adding the results of each column of the dataset
targetTotal <- sapply(as.data.frame(chunksSummary), sum)

# Returning the results obtained after the sum of all values
# absolute values
print(targetTotal)
# Proportional values
print(prop.table(targetTotal))

# Summarized final proportion: 
# No          Yes 
# 180827810   456845
# 0.997479958 0.002520042

# Plotting the results obtained in the target variable

data.frame(downloadedApp = c('No','Yes'), counts = targetTotal) %>%
  ggplot(aes(x = downloadedApp, y = counts, fill = downloadedApp)) +
  geom_bar(stat = 'identity') +
  labs(title = 'Proportion of downloaded Apps by users') +
  theme_bw()

#-------------Balancing Training Dataset---------------#

# Clearly the proportion of the target variable data is unbalanced
# which would make harder the learning model process for cases where users have downloaded
# We will correct this distortion by creating a new dataset

# Extracting from each part of the dataset all records downloaded ('is_attributed == 1')

# Function to load data, remove duplicate lines and capture users who have downloaded
sapply(chunks, function(c) {
  chunk <- fread(c, col.names = colNames)
  chunk <- chunk[!duplicated(chunk), ]
  fwrite(chunk[chunk$is_attributed == 1,], 'train_sample.csv', append = TRUE)
  return('Saved file!')
})

# Summarizing the number of lines in which is_attributed == 1
usersYes <- t(targetTotal)[2]

# Definition of the number of lines that should be sampled in each chunk for users who have not downloaded
usersSampleNo <- round(usersYes / length(chunks))

# Applying the seed for sampling reproduction
set.seed(42)

# sampling in each chunk lines which the class is_attributed == 0

## Function to load the data, remove duplicate lines and capture users who did not download (by sampling)
sapply(chunks, function(c) {
  chunk <- fread(c, col.names = colNames)
  chunk <- chunk[!duplicated(chunk), ]
  chunk <- chunk[chunk$is_attributed == 0, ]
  fwrite(chunk[sample(1:nrow(chunk), usersSampleNo),], 'train_sample.csv', append = TRUE)
  return('Saved file!')
})

# Loading the dataset
dataTarget = fread('train_sample.csv')

# Plotting the obtained data in a graph

dataTarget %>%
  mutate(downloadedApp = factor(is_attributed, labels = c('No', 'Yes'))) %>%
  ggplot(aes(x = downloadedApp, fill = downloadedApp)) +
  geom_bar() +
  labs(title = 'Proportion of downloaded Apps by users') +
  ylab('count') +
  theme_bw()

#-----------------Loading Test Dataset-----------------#

# We will apply the same solution adopted for the training data
# We will do this to have a better proportion when we apply the training/testing phase of the model

# defining test chunks
numTestChunks <- 5
totalTestLines <- nlines('test.csv') - 1
maxLinesTestChunks <- round(totalTestLines / numTestChunks)

# In the same way with the training data, we will use the split command (Linux) to do the division
# of test data in 5 chunks -> total volume per chunk of 3,758,094 lines
# drgm@sandbox:~/Big_Data_Analytics_R/Project_Feedback$ split test.csv -l 3758094

# determining chunk names with test dataset
testChunks <- c('test_part1.csv', 'test_part2.csv', 'test_part3.csv', 'test_part4.csv', 'test_part5.csv')

# loading the first test data chunks
testChunk <- fread(testChunks[1])

# checking firt lines
head(testChunk)

#-------------------Exploratory Analysis-------------------#

# Checking the data structure in the dataset
str(dataTarget)
View(dataTarget)

# checking the existence of missing values in the dataset
sapply(dataTarget, function(v) {anyNA(v)})

# only attributed_time has NA`s values
# let's count how many NA's values we have in the "attributed_time" column
sum(is.na(dataTarget$attributed_time))

# we have 456845 "NA" records in the dataset
# we conclude that these records are related to the fact that the user has not downloaded
# it makes perfect sense not to have a record of an action that was not performed by the user

# let's check unique records of int variables (ip, app, device, os, channel)
dataTarget %>%
  select('ip') %>%
  n_distinct()
dataTarget %>%
  select('app') %>%
  n_distinct()
dataTarget %>%
  select('device') %>%
  n_distinct()
dataTarget %>%
  select('os') %>%
  n_distinct()
dataTarget %>%
  select('channel') %>% 
  n_distinct()

# ip: 253051
# app: 336
# device: 1881
# OS: 190
# channel: 180

# by the number of ips mapped, it is possible to conclude that there can be multiples
# ads associated with the registration of a single user

# Confirming that the target variable is represented by only two different classes
dataTarget %>%
  select('is_attributed') %>% 
  n_distinct()

#----------------Data Munging----------------#

# attributed_time for is_attributed = 0 -> NA
# data strucutre of attributed_time e click_time -> POSIXct


#--------------------------------------------#

# Evaluating the click time variable
# Checking the distribution of dates when the clicks occurred
summary(dataTarget$click_time)

# checking the time period in which the clicks occurred
max(dataTarget$click_time) - min(dataTarget$click_time)
# Period of clicks occurred in a period of approximately 3 days

# We will create a time series graph to check the number of ad clicks per hour
# and which led to the downloading of apps during the entire time span
dataTarget %>%
  mutate(datesFix = floor_date(click_time, unit = 'hour')) %>%
  group_by(datesFix) %>%
  summarise(downloadsDone = sum(is_attributed)) %>%
  ggplot(aes(x = datesFix, y = downloadsDone)) +
  geom_line() +
  scale_x_datetime(date_breaks = '4 hours', date_labels = '%d %b /%H hrs') +
  theme_light() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  xlab('Clicks done by users') +
  ylab('Downloads') +
  labs(title = 'Downloads done by period of time')

# The series shows that there is consistency of downloads throughout the day within the 3-day period
# downloads peak occurred from 00:00 to 13:00 hrs

# We will do the same analysis considering clicks that did not lead to downloads
dataTarget %>%
  mutate(datesFix = floor_date(click_time, unit = 'hour')) %>%
  group_by(datesFix) %>%
  summarise(notDownloaded = sum(!is_attributed)) %>%
  ggplot(aes(x = datesFix, y = notDownloaded)) +
  geom_line() +
  scale_x_datetime(date_breaks = '4 hours', date_labels = '%d %b /%H hrs') +
  theme_light() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  xlab('Clicks done by users') +
  ylab('Not downloaded') +
  labs(title = 'Downloads not done by period of time')

# The periods in which clicks occurred and no downloads were made follow a distribution
# similar to the previous plot

# Let's overlap the two plots

dataTarget %>%
  mutate(datesFix = floor_date(click_time, unit = 'hour')) %>%
  group_by(datesFix) %>%
  summarise(downloadsDone = sum(is_attributed),
            notDownloaded = sum(!is_attributed)) %>%
  ggplot() +
  geom_line(aes(x = datesFix, y = downloadsDone, color = 'Yes')) +
  geom_line(aes(x = datesFix, y = notDownloaded, color = 'No')) +
  scale_x_datetime(date_breaks = '4 hours', date_labels = '%d %b /%H hrs') +
  theme_light(base_size = 20) +
  theme(axis.text.x = element_text(angle = 45, hjust =1)) +
  xlab('Clicks done by users') +
  ylab('Clicks Attributed') +
  labs(title = 'Cross Evaluation Downloaded x not Downloaded', colour = 'Downloaded')

#-------------------Feature Engineering---------------#

# Let's prepare the training data for creating the predictive model:
# The attributed_time variable will be removed as it has a direct relationship with what we want to predict in the target variable (is_attributed = 1)
# We will transform the click_time variable to which we will remove the "Month" unit, as it is not relevant in this sample
# Let's group the ip variable with the other variables of the dataset due to the large amount of unique values it has
dataTarget <- dataTarget %>%
  select(-c(attributed_time)) %>%
  mutate(day = day(click_time), hour = hour(click_time)) %>%
  select(-c(click_time)) %>%
  add_count(ip, day, hour) %>% rename('ip_day_hour' = n) %>%
  add_count(ip, hour, app) %>% rename('ip_hour_APP' = n) %>%
  add_count(ip, hour, device) %>% rename('ip_hour_DEVICE' = n) %>%
  add_count(ip, hour, os) %>% rename('ip_hour_OS' = n) %>%
  add_count(ip, hour, channel) %>% rename('ip_hour_CHANNEL' = n) %>%
  select(-c(ip))

# The same feature engineering process will be applied to each chunk of the test dataset
colTestNames <- colnames(testChunk)
sapply(testChunks, function(x) {
  testChunk <- fread(x, col.names = colTestNames)
  testChunk <- testChunk %>%
    mutate(day = day(click_time), hour = hour(click_time)) %>%
    select(-c(click_time)) %>%
    add_count(ip, day, hour) %>% rename('ip_day_hour' = n) %>%
    add_count(ip, hour, app) %>% rename('ip_hour_APP' = n) %>%
    add_count(ip, hour, device) %>% rename('ip_hour_DEVICE' = n) %>%
    add_count(ip, hour, os) %>% rename('ip_hour_OS' = n) %>%
    add_count(ip, hour, channel) %>% rename('ip_hour_CHANNEL' = n) %>%
    select(-c(ip))
  fwrite(testChunk, paste('modif_', x, sep = ''), append = TRUE)
  return('Test chunk saved!')
})

# We will convert the target variable from int type to factor type because we will solve a problem where our target variable
# will result in a value of type categorical
dataTarget$is_attributed <- as.factor(dataTarget$is_attributed)

# Saving the result in a consolidated file
fwrite(dataTarget, 'dataTarget.csv')
str(dataTarget)

###########-------------------#############

# to "push back" the dataTarget dataset, if necessary
dataTarget <- fread('dataTarget.csv')

###########-------------------#############

# Assessing the importance of each variable
# We will use the RandomForest algorithm to evaluate the weight that each variable has in the prediction of the target variable
# A range for defining the hyperparameter values will be defined for further evaluation
# about which configuration is most efficient

# defining a dataframe with the results of configuration evaluations
featureEvalRF <- data.frame()

# establishing the range of trees and nodes
RFTrees <- 1:25
RFNodes <- 1:10

# number of models to be created
modelComb <- length(RFTrees) * length(RFNodes) 

# let's create a variable so that we can follow the evaluation in progress
# then the most important attributes will be evaluated and the confusion matrix will be generated
count <- 0
for(t in RFTrees) {
  for(n in RFNodes) {
    set.seed(100)
    modelEvalRF <- randomForest(is_attributed ~ .,
                                data = dataTarget,
                                ntree = t,
                                nodesize = n,
                                importance = TRUE)
    confusionRF <- confusionMatrix(table(
      data = modelEvalRF$y,
      reference = modelEvalRF$predicted
    ))
    
    featureEvalRF <- rbind(featureEvalRF, data.frame(
      nodes = n,
      trees = t,
      accuracy = unname(confusionRF$overall['Accuracy'])
    ))
    
    count <- count + 1
  }
}

# The results will be saved in a .csv file
fwrite(featureEvalRF, 'featureEvalRF.csv')

# loading the dataframe with the results obtained
featureEvalRF <- fread('featureEvalRF.csv')

# Returning the configuration that presented the best performance
bestFeatureRF <- featureEvalRF[featureEvalRF$accuracy == max(featureEvalRF$accuracy),]
bestFeatureRF

#    nodes trees  accuracy
# 1:    10    25 0.9185053

modelRF <- randomForest(is_attributed ~ .,
                        data = dataTarget,
                        ntree = bestFeatureRF$trees,
                        nodesize = bestFeatureRF$nodes,
                        importance = TRUE)

# Returning the model results
modelRF

# Type of random forest: classification
# Number of trees: 25
# No. of variables tried at each split: 3

# OOB estimate of  error rate: 8.12%
# Confusion matrix:
#        0      1 class.error
# 0 440114  16729  0.03661871
# 1  57425 399419  0.12569936

# We will create a graph to visualize the importance level of each variable in determining the target variable
varEval <- as.data.frame(varImpPlot(modelRF))

# returning a list with the importance level of each variable in descending order
varEval[order(varEval$MeanDecreaseAccuracy, decreasing = TRUE),]

#                 MeanDecreaseAccuracy MeanDecreaseGini
# app                        38.517955       194270.067
# hour                       24.078986         7547.198
# day                        23.425862         2150.136
# ip_day_hour                19.625139         4225.207
# channel                    16.329030        90920.962
# ip_hour_APP                14.782068         2674.303
# ip_hour_CHANNEL            12.496946         7251.455
# ip_hour_DEVICE             11.885089        11480.762
# ip_hour_OS                 11.423202         2516.927
# device                      5.617553        22971.100
# os                          4.337295        18479.965

#------------------Creating the predictive model-----------------#

# We will pick 3 algorithms in order to train the predicitive model
# which best fit very large datasets:

# Naive-Bayes
# Random Forest
# Adaboost

#---------------Naive-Bayes Algorithm------------#

# creating the model with Naive-Bayes
modelNB <- naiveBayes(is_attributed ~ ., data = dataTarget)

# making predictions
predictionsNB <- predict(modelNB, dataTarget, type = c('class', 'raw'))

# Confusion Matrix
confusionMatrix(table(pred = predictionsNB, data = dataTarget$is_attributed))

#################################

# Accuracy : 0.648  

# pred      0      1
#    0 440345 305082
#    1  16500 151763

# 'Positive' Class : 0 

#################################

# Generating the ROC curve
rocNB <- prediction(as.numeric(predictionsNB), dataTarget$is_attributed)
perfNB <- performance(rocNB, 'tpr', 'fpr')
plot(perfNB, col = 'blue', main = 'Curva ROC - Naive-Bayes')
abline(a = 0, b = 1)

# Precision/Recall
prNB <- performance(rocNB, 'prec', 'rec')
plot(prNB, main = 'Curva Precision/Recall - Naive-Bayes')


#---------------Random Forest algorithm--------------#

# Creating the model with Random Forest
modelAppRF <- randomForest(is_attributed ~ .,
                           ntree = 25,
                           nodesize = 10,
                           data = dataTarget)

# Making Predictions
predictionsAppRF <- predict(modelAppRF, dataTarget, type = 'response')

# Confusion Matrix
confusionMatrix(table(pred = predictionsAppRF, data = dataTarget$is_attributed))

###################################

# Accuracy : 0.929  

#     data
# pred      0      1
#    0 444632  52686
#    1  12213 404159

# 'Positive' Class : 0 

###################################

# Generating ROC Curve
rocRF <- prediction(as.numeric(predictionsAppRF), dataTarget$is_attributed)
perfRF <- performance(rocRF, 'tpr', 'fpr')
plot(perfRF, col = 'green', main = 'Curva ROC - Random Forest')
abline(a = 0, b = 1)

# Precision/Recall
prRF <- performance(rocRF, 'prec', 'rec')
plot(prRF, main = 'Curva Precision/Recall - Random Forest')

str(predictionsAppRF)

#---------------AdaBoost Algorithm------------#

# Creating the model with AdaBoost
modelAdaB <- adaboost(is_attributed ~ ., data = as.data.frame(dataTarget), nIter = 10, method = 'adaboost')

# Making predictions
predictionsAdaB <- predict(modelAdaB, dataTarget, type = 'class')

# Confusion Matrix
confusionMatrix(table(pred = predictionsAdaB$class, data = dataTarget$is_attributed))

######################

# Accuracy :  0.9569

# pred      0      1
#    0 436693  19221
#    1  20152 437624

# 'Positive' Class : 0 

######################

# Generating ROC Curve
rocAdaB <- prediction(as.numeric(predictionsAdaB$class), dataTarget$is_attributed)
perfAdaB <- performance(rocAdaB, 'tpr', 'fpr')
plot(perfAdaB, main = 'Curva ROC - AdaBoost')
abline(a = 0, b = 1)

# Precison/Recall
prAdaB <- performance(rocAdaB, 'prec', 'rec')
plot(prAdaB, main = 'Curva Precision/Recall - AdaBoost')

#-----------------Optimization of the best performing model--------------#

# We will seek to optimize the performance of the model developed with AdaBoost by changing some parameters
modelOptAda <- adaboost(is_attributed ~ .,
                         data = as.data.frame(dataTarget),
                         nIter = 50,
                         method = 'adaboost')

# Making predictions with the optimized model
predictionsOptAda <- predict(modelOptAda, dataTarget, type = 'class')

confusionMatrix(table(pred = predictionsOptAda$class, data = dataTarget$is_attributed))

######################

# Accuracy : 0.9617  

# pred      0      1
#    0 429889   8011
#    1  26956 448834

######################

# Generating ROC Curve
rocOptAdaB <- prediction(as.numeric(predictionsOptAda$class), dataTarget$is_attributed)
perfOptAdaB <- performance(rocOptAdaB, 'tpr', 'fpr')
plot(perfOptAdaB, main = 'Curva ROC - AdaBoost Optimized')
abline(a = 0, b = 1)

# Precison/Recall
prOptAdaB <- performance(rocOptAdaB, 'prec', 'rec')
plot(prOptAdaB, main = 'Curva Precision/Recall - AdaBoost Optimized')


#-------------------Predictions with test dataset--------------------#

# We will make the predictions using the model with Optimized Adaboost
# The results obtained will be saved in a .csv file

# Calling some variables already used and that were modified due to transformations in the test dataset
testModifChunks <- c('modif_test_part1.csv', 'modif_test_part2.csv', 'modif_test_part3.csv', 
                     'modif_test_part4.csv', 'modif_test_part5.csv')
testModifChunk <- fread(testModifChunks[1])
colModifTestNames <- colnames(testModifChunk)

sapply(testModifChunks, function(x) {
  chunk <- fread(x, col.names = colModifTestNames)
  predictionsTest <- predict(modelOptAda, chunk[,!'click_id'], type = 'class')
  fwrite(data.frame(click_id = as.integer(chunk$click_id),
                    is_attributed = as.numeric(predictionsTest$class)), 'predictions.csv', append = TRUE)
  rm(chunk, predictionsTest)
  return('Saved results!')
})

