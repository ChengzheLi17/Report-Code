
getwd()
install.packages("corrplot")
install.packages("randomForest")
install.packages("ROCR")
install.packages("pROC")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("caret")
install.packages("plyr")

library(dplyr)
library("corrplot")
library(randomForest)
library(pROC)
library(ggplot2)
library("caret")
library(ROCR)
library(plyr)


#Import data
data <- read.csv("~/Desktop/Classification/bank_personal_loan.csv")
#data summary
summary(data)

#Check for missing values
sum(is.na(data))
sum(complete.cases(data))

#delete ZIP.Code
data <- select(data, -ZIP.Code)

class(data)
mode(data$Online)
#Data distribution
table(data$CreditCard)
table(data$Family)
table(data$Education)
table(data$Personal.Loan)
table(data$Securities.Account)
table(data$CD.Account)
table(data$Online)

#Modify outliers
data$Experience[data$Experience < 0] <- 0

#Visualization
hist(data$Age,
     xlim = c(20,70),
     freq=TRUE)
hist(data$Experience,
     xlim = c(0,50),
     freq=TRUE)
hist(data$Income,
     xlim = c(0,250),
     freq=TRUE)
hist(data$Family,
     xlim = c(1,4),
     freq=TRUE)
hist(data$CCAvg,
     xlim = c(0,10))
table(data$Mortgage)
hist(data$Mortgage,
     xlim = c(0,700))

head(data)
dim(data)
str(data)

#Correlation analysis


cor(data)
corrplot(cor(data),method='color',addCoef.col='grey')

#RandomForest
data1 <- data


#data1$Personal.Loan=as.factor(data1$Personal.Loan)
train_sub = sample(nrow(data1),7/10*nrow(data1))
train_data = data1[train_sub,]
test_data = data1[-train_sub,]



loan_RF <- randomForest(Personal.Loan~.,
                        data = train_data,
                        ntree =820,
                        mtry=6,
                        importance=TRUE ,
                        proximity=TRUE)

print(loan_RF)
importance(loan_RF)

#loan_RF$importance
varImpPlot(loan_RF, main = "variable importance")


#predict for the test set
RFpre_data <- predict(loan_RF,type='response',newdata=test_data)
#ROC
Roc <- roc(test_data$Personal.Loan,as.numeric(RFpre_data))
plot(Roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), 
     max.auc.polygon=TRUE,
     auc.polygon.col="skyblue",
     print.thres=TRUE,
     main='RandomForest-ROC,mtry=6,ntree=820')
MDSplot(loan_RF,data1$Personal.Loan,k=2,palette=NULL,pch=20)
plot(loan_RF)
treesize(loan_RF,terminal = TRUE)


predict =ifelse(RFpre_data>0.5,1,0)
test_data$predict=predict
true_value=test_data[,8]
predict_value=test_data[,13]
error = predict_value-true_value
accuracy = (nrow(obs_p_t)-sum(abs(error)))/nrow(obs_p_t)
precision=sum(true_value & predict_value)/sum(predict_value)
recall=sum(predict_value & true_value)/sum(true_value)
F_measure=2*precision*recall/(precision+recall)   
print(accuracy)
print(precision)
print(recall)
print(F_measure)
table(true_value,predict_value) 

#10 Folds RandomForest
data2 <- data
set.seed(9)
folds <- createFolds(y=data2$Personal.Loan,k=10)
max=0
num=0
for(i in 1:10){    
  fold_test <- data2[folds[[i]],]   #put folds[[i]] as test set  
  fold_train <- data2[-folds[[i]],]   # The rest is the training set    
  print("***group number***")   
  fold_pre <-randomForest(Personal.Loan~.,
                          data= fold_train,
                          mtry=6,
                          ntree=820, 
                          proximity=TRUE,
                          importance=TRUE)
  fold_predict <- predict(fold_pre,type='response',newdata=fold_test) 
  fold_predict =ifelse(fold_predict>0.5,1,0)  
  fold_test$predict = fold_predict  
  fold_error = fold_test[,13]-fold_test[,8]  #True Value-13column,predict value-24value
  fold_accuracy = (nrow(fold_test)-sum(abs(fold_error)))/nrow(fold_test)   
  print(i)  
  print("***accuracy of test set***") 
  print(fold_accuracy)  
  print("***accuracy of train set***")
  fold_predict2 <- predict(fold_pre,type='response',newdata=fold_train)  
  fold_predict2 =ifelse(fold_predict2>0.5,1,0)  
  fold_train$predict = fold_predict2  
  fold_error2 = fold_train[,13]-fold_train[,8]  
  fold_accuracy2 = (nrow(fold_train)-sum(abs(fold_error2)))/nrow(fold_train)   
  print(fold_accuracy2)      
  if(fold_accuracy>max)    {    
    max=fold_accuracy      
    num=i   
  }  }
print(max)
print(num)
#Choose the sample with the highest accuracy
#test set
testi <- data2[folds[[num]],]
#train
traini <- data2[-folds[[num]],]
#new model
prei <- randomForest(Personal.Loan~.,
                     data= traini,
                     ntree=820,
                     mtry=6,
                     proximity=TRUE,
                     importance=TRUE)

#check the importance of each factor
importance<-importance(x= prei)
importance
set.seed(100)
varImpPlot(prei)

#train test set
predicti <- predict (prei,type='response',newdata=testi)
testi$predict = predicti
#train train set
predicti2 <- predict (prei,type='response',newdata=traini)
traini$predict = predicti2
#train the whole data
predicti3 <- predict (prei,type='response',newdata=data2)
data2$predict = predicti3


#ROC curve

library(ROCR)
#train set：     
pred <- prediction(predicti2,traini$Personal.Loan)     
performance(pred,'auc')@y.values        #AUC value
perf <- performance(pred,'tpr','fpr')
plot(perf,col="red")
#test set
pred2 <- prediction(predicti,testi$Personal.Loan)   
performance(pred2,'auc')@y.values      
perf2 <- performance(pred2,'tpr','fpr')
plot(perf2,add=TRUE, col ="blue")#add即为加在上一个ROC图上面（训练）
#all data
pred3 <- prediction(predicti3,data2$Personal.Loan)   
performance(pred3,'auc')@y.values       
perf3 <- performance(pred3,'tpr','fpr')
plot(perf3,add=TRUE, col ="green")


#confusion matrix
#train set
predict =ifelse(predicti2>0.5,1,0)
traini$predict = predict
true_value=traini[,8]
predict_value=traini[,13]
#calculate the accuracy
error = predict_value-true_value
accuracy = (nrow(traini)-sum(abs(error)))/nrow(traini)
precision=sum(true_value & predict_value)/sum(predict_value)
recall=sum(predict_value & true_value)/sum(true_value)
F_measure=2*precision*recall/(precision+recall)   
print(accuracy)
print(precision)
print(recall)
print(F_measure)
table(true_value,predict_value)    

#str(traini)
#test set
predict =ifelse(predicti>0.5,1,0)
testi$predict = predict
true_value=testi[,8]
predict_value=testi[,13]
error = predict_value-true_value
accuracy = (nrow(testi)-sum(abs(error)))/nrow(testi)
precision=sum(true_value & predict_value)/sum(predict_value)
recall=sum(predict_value & true_value)/sum(true_value)
F_measure=2*precision*recall/(precision+recall)   
print(accuracy)
print(precision)
print(recall)
print(F_measure)
table(true_value,predict_value)  

#all data
predict =ifelse(predicti3>0.5,1,0)
data2$predict = predict
true_value=data2[,8]
predict_value=data2[,13]
error = predict_value-true_value
accuracy = (nrow(data2)-sum(abs(error)))/nrow(data2)
precision=sum(true_value & predict_value)/sum(predict_value)
recall=sum(predict_value & true_value)/sum(true_value)
F_measure=2*precision*recall/(precision+recall)   
print(accuracy)
print(precision)
print(recall)
print(F_measure)
table(true_value,predict_value)


#10 Folds Logistic
data3 <- data
library(plyr)
library(caret)
logfolds <- createFolds(y=data3$Personal.Loan,k=10)
length(logfolds)

for(i in 1:10){    
  logfold_test <- data3[logfolds[[i]],]    
  logfold_train <- data3[-logfolds[[i]],]   
  print("***group number***")   
  logfold_pre <-glm(Personal.Loan ~.,family=binomial(link = "logit"),data = logfold_train)
  logfold_predict <- predict(logfold_pre,type='response',newdata=logfold_test) 
  logfold_predict =ifelse(logfold_predict>0.5,1,0)  
  logfold_test$predict = logfold_predict  
  logfold_error = logfold_test[,13]-logfold_test[,8]  
  logfold_accuracy = (nrow(logfold_test)-sum(abs(logfold_error)))/nrow(logfold_test)   
  print(i)  
  print("***accuracy of test set***") 
  print(logfold_accuracy)  
  print("***accuracy of train set***")
  logfold_predict2 <- predict(logfold_pre,type='response',newdata=logfold_train)  
  logfold_predict2 =ifelse(logfold_predict2>0.5,1,0)  
  logfold_train$predict = logfold_predict2  
  logfold_error2 = logfold_train[,13]-logfold_train[,8]  
  logfold_accuracy2 = (nrow(logfold_train)-sum(abs(logfold_error2)))/nrow(logfold_train)   
  print(logfold_accuracy2)      
}
