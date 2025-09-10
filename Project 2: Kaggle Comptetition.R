#### Kaggle competition ####

library(ggcorrplot)
library(mlrMBO)
library(nnet)
library(caret)
library(xgboost)
library(randomForest)

#### Dati #### 

data <- read.csv("train.csv")
summary(data)
df <- data[,c(2:20)] 
df$class <- as.factor(data$Class)
df <- df[df$quality == 1,] #TOLGO QUALITY = 0
df <- df[,2:ncol(df)]

test.import <- read.csv("test.csv")

## Correlazione
corr <- round(cor(df[,1:18]),3)
ggcorrplot(corr, title = "Matrice di correlazione",
           type = "upper", lab = T, colors = c("indianred1","white","indianred1")) +
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        axis.text.x = element_text(angle = 90))

df <- df[,c("pre_screening","ma1","ma3","ma6",
            "exudate1","exudate4","exudate6","exudate8",
            "macula_opticdsc_distance","opticdisc_diameter",
            "am_fm_classification","class")]

test.FINAL <- test.import[,c("pre_screening","ma1","ma3","ma6",
                             "exudate1","exudate4","exudate6","exudate8",
                             "macula_opticdsc_distance","opticdisc_diameter",
                             "am_fm_classification")]

## Bilanciamento
mean(as.numeric(df$class)-1) #Bilanciato

## Istogrammi

for (i in 2:(ncol(df)-2)) {
  hist(df[,i], main = colnames(df)[i])
}

for (i in 2:(ncol(df)-2)) {
  boxplot(df[,i], main = colnames(df)[i])
}



#### Training e test ####

set.seed(123)
indexes <- sample(1:nrow(df), nrow(df)*0.8)
training <- df[indexes,]
test <- df[-indexes,]

#Task comune
task <- makeClassifTask(data = training, target = "class")



#### Testati KNN, SVM, DT -> risultati non soddisfacenti



#### Consegna 1 e 2: NN ####

## Cross Validation mlrmbo
set.seed(123)
par.set.NN <- makeParamSet(
  makeIntegerParam("size", lower = 3, upper = 15),
  makeNumericParam("decay", lower = -3, upper = -1, trafo = function(x) 10^x))
ctrl.NN <- makeMBOControl()
ctrl.NN <- setMBOControlTermination(ctrl.NN, iters = 30)
tune.ctrl.NN <- makeTuneControlMBO(mbo.control = ctrl.NN)
learner.NN <- makeLearner("classif.nnet", trace = F, maxit = 1000)
run.NN <- tuneParams(learner.NN, task, cv3,
                     measures = f1, par.set = par.set.NN,
                     control = tune.ctrl.NN, show.info = T)

mod.NN <- nnet(class ~ ., data = training, maxit = 1000,
               size = run.NN$x$size, 
               decay = run.NN$x$decay)
confusionMatrix(as.factor(predict(mod.NN, type = "class")), training$class)

## Test
preds.NN <- as.factor(predict(mod.NN, test, type = "class"))
confusionMatrix(preds.NN, test$class)

## Allenamento su tutto il training
mod.FINAL.1 <- nnet(class ~ ., data = df, maxit = 1000,
                    size = run.NN$x$size, 
                    decay = run.NN$x$decay)
confusionMatrix(as.factor(predict(mod.FINAL.1, type = "class")), df$class)

## Previsioni
preds.FINAL.1 <- as.numeric(predict(mod.FINAL.1, test.FINAL, type = "class"))

consegna.1 <- cbind(test.import$ID, preds.FINAL.1)
colnames(consegna.1) <- c("ID","Class")
write.csv(consegna.1, "Consegna_Nobili.csv", row.names = F)
#write.csv(consegna.1, "Consegna_Nobili_2.csv", row.names = F)



#### Consegna 3: Ensemble XGBoost, NN, RF ####

### XGBoost

## Cross Validation mlrmbo
set.seed(123)
par.set.XG <- makeParamSet(
  makeNumericParam("eta", lower = -5, upper = -2, trafo = function(x) 10^x),
  makeNumericParam("gamma", lower = -7, upper = -3, trafo = function(x) 10^x),
  makeIntegerParam("max_depth", lower = 3, upper = 15),
  makeIntegerParam("nrounds", lower = 1, upper = 10))
ctrl.XG <- makeMBOControl()
ctrl.XG <- setMBOControlTermination(ctrl.XG, iters = 50)
tune.ctrl.XG <- makeTuneControlMBO(mbo.control = ctrl.XG)
learner.XG <- makeLearner("classif.xgboost", objective = "binary:logistic")
run.XG <- tuneParams(learner.XG, task, cv3,
                     measures = f1, par.set = par.set.XG,
                     control = tune.ctrl.XG, show.info = T)

mod.XG <- xgboost(data = as.matrix(training[,1:11]), label = as.matrix(training$class),
                  eta = run.XG$x$eta, gamma = run.XG$x$gamma,
                  max_depth = run.XG$x$max_depth,
                  nrounds = run.XG$x$nrounds, objective = "binary:logistic")
confusionMatrix(as.factor(as.numeric(
  predict(mod.XG, newdata = as.matrix(training[,1:11])) > 0.5)),
  training$class)

## Test
preds.XG <- as.factor(as.numeric(predict(mod.XG, newdata = as.matrix(test[,1:11])) > 0.5))
confusionMatrix(preds.XG, test$class)

## Allenamento su tutto il training
mod.FINAL.2 <- xgboost(data = as.matrix(df[,1:11]), label = as.matrix(df$class),
                       eta = run.XG$x$eta, gamma = run.XG$x$gamma,
                       max_depth = run.XG$x$max_depth,
                       nrounds = run.XG$x$nrounds, objective = "binary:logistic")
confusionMatrix(as.factor(as.numeric(
  predict(mod.FINAL.2, newdata = as.matrix(df[,1:11])) > 0.5)),
  df$class)

## Previsioni
preds.FINAL.XG <- as.numeric(
  predict(mod.FINAL.2, newdata = as.matrix(test.FINAL)) > 0.5)

### RF

## Cross Validation mlrmbo
set.seed(123)
par.set.RF <- makeParamSet(
  makeIntegerParam("ntree", lower = 10, upper = 500),
  makeIntegerParam("mtry", lower = 3, upper = 8),
  makeIntegerParam("nodesize", lower = 10, upper = 30))
ctrl.RF <- makeMBOControl()
ctrl.RF <- setMBOControlTermination(ctrl.RF, iters = 30)
ctrl.RF <- setMBOControlInfill(ctrl.RF, opt.focussearch.points = 20)
tune.ctrl.RF <- makeTuneControlMBO(mbo.control = ctrl.RF)
run.RF <- tuneParams(makeLearner("classif.randomForest"), task, cv3,
                     measures = f1, par.set = par.set.RF,
                     control = tune.ctrl.RF, show.info = T)

mod.RF <- randomForest(class ~ ., data = training,
                       ntree = run.RF$x$ntree,
                       mtry = run.RF$x$mtry,
                       nodesize = run.RF$x$nodesize)
confusionMatrix(predict(mod.RF), training$class)

#Test
preds.RF <- predict(mod.RF, test)
confusionMatrix(preds.RF, test$class)

## Allenamento su tutto il training
mod.FINAL.3 <- randomForest(class ~ ., data = df,
                            ntree = run.RF$x$ntree,
                            mtry = run.RF$x$mtry,
                            nodesize = run.RF$x$nodesize)
confusionMatrix(predict(mod.FINAL.3), df$class)

## Previsioni
preds.FINAL.RF <- as.numeric(predict(mod.FINAL.3, test.FINAL, type = "class"))-1

### Ensemble

ensemble <- cbind(preds.FINAL.1,preds.FINAL.XG,preds.FINAL.RF)
barplot(table(rowMeans(ensemble)), horiz = T)

pred.ENS <- ifelse(rowMeans(ensemble)>0.5, 1, 0)

consegna.2 <- cbind(test.import$ID, pred.ENS)
colnames(consegna.2) <- c("ID","Class")
write.csv(consegna.2, "Consegna_Nobili_3.csv", row.names = F)


