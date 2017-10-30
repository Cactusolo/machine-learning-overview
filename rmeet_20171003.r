download.file("http://kokbent.gitlab.io/rmeet20171003.zip", "rmeet.zip")
unzip("rmeet.zip")

## Titanic
library(tidyverse)
library(titanic)
library(Amelia)

titanic <- titanic_train

rnum <- sample(1:nrow(titanic), size = 200)
test <- titanic[rnum,]
train <- titanic[-rnum,]

mod1 <- glm(Survived ~ Pclass + Sex + SibSp + Parch + Fare, data = train)
pred <- predict(mod1, test, type="response")
pred <- ifelse(pred > 0.5, 1, 0)
table(test$Survived, pred, useNA = "ifany")
mean(pred == test$Survived)

# Imputation first
titanFeature <- titanic %>%
  select(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
impute <- amelia(titanFeature, noms = c("Sex", "Embarked"))
titanImp <- impute$imputations$imp3
titanImp$Survived <- titanic$Survived

test <- titanImp[rnum,]
train <- titanImp[-rnum,]

mod1 <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare, data = train)
pred <- predict(mod1, test, type="response")
pred <- ifelse(pred > 0.5, 1, 0)
table(test$Survived, pred, useNA = "ifany")
mean(pred == test$Survived)

## Proper script for digit recognition
library(tidyverse)
library(nnet)

train <- read.csv("recognition/train.csv")
test <- read.csv("recognition/test.csv")
train$label <- as.factor(train$label)
test$label <- as.factor(test$label)

ind <- sample(1:nrow(train), 1)
select <- train[ind, -1] %>% as.numeric
mat <- matrix(select, ncol=28, nrow=28, byrow=F)
image(mat, col=gray.colors(255))
image(mat[,nrow(mat):1], col=gray.colors(255))

colName <- colnames(train)[-1]
formula.1 <- paste(colName, collapse = "+")
formula <- paste("label~", formula.1, sep="") %>%
  as.formula

weights <- read.csv("recognition/weights.csv")
weights <- weights$x

fitNN <- nnet(formula, data = train, Wts = weights,
              size = 20, decay = 0.1, maxit = 10, MaxNWts = 20000)
pred <- predict(fitNN, test, type="class")
mean(pred == test$label)
table(test$label, pred)

fitRF <- randomForest(train[,-1], train$label, keep.forest = T)
predRF <- predict(fitRF, test[,-1])
table(test$label, predRF)

## K means SNP100
set.seed(1234)
snp <- read.csv("snp100/snp100_2017.csv", sep=";", as.is = T)

km <- kmeans(snp[,c(-1, -2)], centers = 5, nstart = 20)
km

for (i in 1:5) {
  cl <- which(km$cluster == i)
  cl <- snp$Name[cl]
  print(paste("Cluster ", i, sep=""))
  print("========================================")
  print(cl)
  print(" ")
}
