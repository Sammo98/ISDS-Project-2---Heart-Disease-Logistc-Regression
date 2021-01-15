              ##### INSTALLING PACKAGES AND IMPORTING DATA

#install.packages('tidyverse')
#install.packages('corrplot')
#install.packages('ggplot2')
#install.packages("caret")
#install.packages("lessR") 
#install.packages("ROCR")
library(tidyverse)
library(corrplot)
library(ggplot2)
library(caret)
library(lessR)
library(pROC)

# IMPORTING DATA

#url:https://archive.ics.uci.edu/ml/datasets/heart+Disease

#Saved as newdata.csv on my PC

df <- read.csv(file="~/Desktop/newdata.csv")


                      ##### EXPLORATORY ANALYSIS

###CLEANING DATA

sum(is.na(df)) # No Rows are missing values
duplicated(df) # Row 165 is duplicated
df <- df[-c(165),] 
duplicated(df) # Now there are no duplicated rows



#Changing Names of Variables for Clarity

df$sex[df$sex==1] = "M"
df$sex[df$sex==0] = "F"

df$target[df$target==1] = "Healthy"
df$target[df$target==0] = "Unhealthy"


# Changing discrete variables to factors

df$sex = as.factor(df$sex)
df$cp = as.factor(df$cp)
df$fbs = as.factor(df$fbs)
df$restecg = as.factor(df$restecg)
df$exang = as.factor(df$exang)
df$slope = as.factor(df$slope)
df$ca = as.integer(df$ca)
df$ca = as.factor(df$ca)
df$thal = as.integer(df$thal)
df$thal = as.factor(df$thal)
df$target = as.factor(df$target)

#Changing Integer Variables to Numeric Variables
df$age = as.numeric(df$age)
df$trestbps = as.numeric(df$trestbps)
df$chol = as.numeric(df$chol)
df$thalach = as.numeric(df$thalach)

str(df) #All variables are now the correct structure

# CA(93, 159, 164, 251) AND THAL (49&281) MISSING - SET VALUES AS MODE

col_thal = df$thal
col_thal[49] = 2
col_thal[281] = 2
df$thal <- col_thal
df$thal <- factor(df$thal)

col_ca = df$ca
col_ca[93] = 0
col_ca[159] = 0
col_ca[164] = 0
col_ca[251] = 0

df$ca <- col_ca
df$ca <- factor(df$ca)



##Graphical Representation of Variables and Numerical Data


# Response Variable - target

ggplot(df, aes(x=target)) + geom_bar(col="black", fill="cornflowerblue")+
  geom_text(stat="count",aes(label=..count..),
  position=position_stack(0.5), color="white", size=5)



## Discrete Predictor Variables: sex, cp, fbs, restecg, exang, slope, ca, thal
# Replace predictor variable to obtain graphs and xtabs as necessary

#Number in each factor level

xtabs(~cp, data=df) 

#Barchart

ggplot(df, mapping = aes(ca, fill = target))+  
  geom_bar(col = 'black')+   
  xlab('Thalium Stress Test Result') + ylab('Frequency')+
  geom_text(stat="count",aes(label=..count..),
            position=position_stack(0.5), color="white", size=3.5)




  
## Continuous Variables: Age, trestbps, chol, thalach, oldpeak 
# Change variables as needed

#Descriptive Stats

summary(df$oldpeak)
sd(df$oldpeak)
cor(df$thalach, df$oldpeak)

#Histogram with Density Overlay

ggplot(df, aes(x = oldpeak)) +
  geom_histogram(aes(y = ..density..), binwidth = 0.5, fill = "cornflowerblue", 
                 col = "black") +
  geom_density() +
  xlab("ST Depression Induced by Exercise Relative to Rest") + ylab("Density")+
  theme_minimal()

#Boxplot

ggplot(df, mapping = aes(target, oldpeak))+  
  geom_boxplot(fill = "cornflowerblue") + 
  xlab("Target")+ ylab('ST Depression Induced by Exercise Relative to Rest')+
  theme_minimal()





                        ##### DATA ANALYSIS



###CROSS VALIDATION PREPARATION - CREATE TEST AND TRAIN SET

#Create index at split location

set.seed(03051998)
index = createDataPartition(df$target, p = 0.8, list=FALSE, times = 1)

#Partition df into train and test

df_train = df[index,]
df_test = df[-index,]



### MODELLING PROCESS AND FINAL MODEL

## Univariate Logistic Regression - Change Predictor as Necessary

Logit(target~age, data = df_train) 

#OR

uni_model_logit = glm(target~thal, data=df_train, family = binomial)
summary(uni_model_logit)

#Box-Tidwell Test -  Change Continuous Predictor as Necessary

Logit(target~thalach + thalach:log(thalach), data = df_train, brief = TRUE)


##Univariate Probit Regression - Change predictor as necessary

probit_model_final = Logit(target ~ age, method = 'polr', 
                           data = df_train)

#or

uni_model_probit <- glm(target ~ thal, family = binomial(link = "probit"), 
                  data = df_train)
  
summary(uni_model_probit)


## Multivariate Logistic Regression

#Variable Selection for Logistic Regression

null = glm(target ~ 1, data=df_train, family = "binomial")
full = glm(target ~ ., data=df_train, family = "binomial")

step(null, scope=formula(full), direction="forward") 
step(full, direction="backward")
step(null, scope = list(upper=full), direction="both") 

##Post Variable Selection Logistic Regression Model:

final_logit_model = Logit(target~ sex+cp+trestbps+
                      exang+oldpeak+
                      slope+ca+thal, data=df_train)

summary(final_logit_model) 


## Probit Regression Model

#Variable Selection for Probit Regression

null = glm(target ~ 1, family = binomial (link = "probit"), data=df_train )
full = glm(target ~ ., family = binomial (link = "probit"), data=df_train )

step(null, scope=formula(full), direction="forward") 
step(full, direction="backward") 
step(null, scope = list(upper=full), direction="both")


#Post Variable Selection Probit Model:

probit_model_final = Logit(target ~ sex + cp + trestbps + chol + 
                           exang + oldpeak + slope + ca + thal,
                           method = 'polr', data = df_train)

summary(probit_model_final) #AIC 182.24 #


##Diagnostics


#remove influential observations 102, 268, 172 due to cooks distance


dim(df_train)
df_train <- df_train[-c(102,268,172),]
dim(df_train)


##Fit Statistics for Comparison

#LOGIT

final_logit_model = Logit(target~ sex+cp+trestbps+
                            exang+oldpeak+
                            slope+ca+thal, data=df_train)

summary(final_logit_model) 

#PROBIT

probit_model_final = Logit(target ~ sex + cp + trestbps + chol + 
                             exang + oldpeak + slope + ca + thal,
                           method = 'polr', data = df_train)

summary(probit_model_final) 



#Chi Square

with(probit_model_final, null.deviance - deviance)
with(probit_model_final, df.null - df.residual)
with(probit_model_final, pchisq(null.deviance - deviance, df.null - df.residual,
                                lower.tail = FALSE))


#pR2


ll.null <- probit_model_final$null.deviance/-2

ll.proposed <- probit_model_final$deviance/-2

r2 <-(ll.null - ll.proposed) / ll.null
r2

1 - pchisq(2*(ll.proposed - ll.null), df = length(probit_model_final$coefficients)-1)


### FINAL MODEL

final_model = Logit(target~ sex+cp+trestbps+
                            exang+oldpeak+
                            slope+ca+thal, data=df_train)
              




                  ##### RESAMPLING AND VALIDATION


#Specify Type of Training Method and # of Folds

ctrlspecs <- trainControl(method = "cv", number = 10,
                          savePredictions = "all",
                          classProbs = TRUE)

#Set Another Seed

set.seed(03051998)

## Specify LR Model

#install.packages('e1071', dependencies=TRUE)

kfold_final_model <- train(target ~ age+sex+cp+trestbps+chol+
                             thalach+exang+oldpeak+slope+ca+thal,
                           data = df_train, method = "glm", family = binomial,
                           trControl = ctrlspecs)



##Final Model Validation

print(kfold_final_model)

#Predict outcome using final_model applied to test set

pred <- predict(kfold_final_model, newdata=df_test)


#Put Predictions into confusion matrix

confusionMatrix(data=pred, df_test$target)


#AUROC

prob = predict(final_logit_model, newdata=df_test, type = "response")
predi = prediction(prob, df_test$target)
perf <- performance(predi, measure = "tpr", x.measure = "fpr")

plot(perf)
auc = performance (predi, measure = 'auc')
auc <- auc@y.values[[1]]
auc


  
###APPENDIX MODEL

simple_model = Logit(target~ age+sex+cp+exang, data=df_train) #75.2%, AIC = 818.79

ctrlspecs <- trainControl(method = "cv", number = 10,
                          savePredictions = "all",
                          classProbs = TRUE)


kfold_simple_model <- train(target ~ age+sex+cp,
                           data = df_train, method = "glm", family = binomial,
                           trControl = ctrlspecs) 

print(kfold_simple_model)

summary(kfold_simple_model)


pred <- predict(kfold_simple_model, newdata=df_test)

confusionMatrix(data=pred, df_test$target)




