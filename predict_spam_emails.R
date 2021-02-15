
library(caret)
library(gains)
library(forecast)
library(dplyr)
library(pROC)

spambase <- read.csv("spambase.csv", stringsAsFactors = TRUE)

dim(spambase) #4601 rows, 58 columns
str(spambase) #shows structures - all numeric

table(spambase$Spam) #2788 0's (not spam), 1813 1's (spam)
#Proportion of emails that are spam
mean(spambase$Spam)

#Rename Variables whose name was changed when loading data 
spambase <- rename(spambase,"C;" = C., "C("= C..1, "C["=C..2,
                   "C!"=C..3, "C$"=C..4, "C#"= C..5)
#show new variable names
t(t(names(spambase)))
#show summary
summary(spambase)


#aggregate data into groups of spam/not spam and show avg for each variable
aggdata <- aggregate(. ~ Spam, spambase, mean)
#calculate the difference of the spam vs not spam mean for each variable
meandiff <- abs(aggdata[1,] - aggdata[2,]) 
sort(meandiff, decreasing=TRUE) #sort variables by highest to lowest avg spam/not spam difference


#partition the data
set.seed(1) #ensure we get the same partitions
train.rows <- sample(rownames(spambase), nrow(spambase)*.6) #random 60% of rows for training data
train.data <- spambase[train.rows, ]
valid.rows <- setdiff(rownames(spambase), train.rows) #other 40% - validation data
valid.data <- spambase[valid.rows, ]


##full model
#logistic regression
logit.reg <- glm(Spam ~ ., data = train.data, family = "binomial")
options(scipen=999)
summary(logit.reg)
#deviance, AIC, BIC
logit.reg$deviance
AIC(logit.reg)
BIC(logit.reg)


#in sample prediction
pred.glm.train <- predict(logit.reg, type = "response")
#out of sample prediction (more important)
pred.glm.valid <- predict(logit.reg, newdata = valid.data, type = "response")


#ROC in_sample prediction
r_insample <- roc(train.data$Spam, pred.glm.train,levels = c(0, 1), direction = "<")
plot.roc(r_insample)
auc(r_insample) 
#ROC out_sample prediction
r_outsample <- roc(valid.data$Spam, pred.glm.valid, levels = c(0, 1), direction = "<")
plot.roc(r_outsample)
auc(r_outsample)

#logit.reg probabilities
df <- data.frame(actual = valid.data$Spam[1:5], predicted = pred.glm.valid[1:5])
df

#logit.reg confusionmatrix
confusionMatrix(as.factor(ifelse(pred.glm.valid >= 0.5, "1", "0")),
                as.factor(valid.data$Spam), positive = "1")

#logit.reg lift chart
gain <- gains(valid.data$Spam, pred.glm.valid, groups = length(pred.glm.valid))

plot(c(0, gain$cume.pct.of.total * sum(valid.data$Spam)) ~ c(0, gain$cume.obs),
     xlab = "# of cases", ylab = "Cumulative", main = "Lift Chart", type = "l")
lines(c(0, sum(valid.data$Spam)) ~ c(0, nrow(valid.data)), lty = 2)

#logit.reg decile chart
gain2 <- gains(valid.data$Spam, pred.glm.valid)
barplot(gain2$mean.resp / mean(df$actual), names.arg = gain2$depth, xlab = "Percentile", 
        ylab = "Mean Response", main = "Decile-wise Lift Chart")

#odds
odds <- data.frame(summary(logit.reg)$coefficient, odds =  exp(coef(logit.reg)))
round(odds, 4)


##reduced model using forward selection

#create null model then use forward selection to find a reduced model
glm.null <- glm(Spam  ~ 1, data = train.data, family = "binomial")
glm.fwd <- step(glm.null, scope = list(glm.null, upper = logit.reg), direction = "forward")
summary(glm.fwd)
#fwd model deviance, AIC, BIC
glm.fwd$deviance
AIC(glm.fwd)
BIC(glm.fwd)

#glm.fwd out of sample prediction
pred.fwd.valid <- predict(glm.fwd, valid.data)

#glm.fwd confusionmatrix
confusionMatrix(as.factor(ifelse(pred.fwd.valid >= 0.5, "1", "0")),
                as.factor(valid.data$Spam), positive = "1")
#glm.fwd lift chart
gainfwd <- gains(valid.data$Spam, pred.fwd.valid, groups = length(pred.fwd.valid))
plot(c(0, gainfwd$cume.pct.of.total * sum(valid.data$Spam)) ~ c(0, gainfwd$cume.obs),
     xlab = "# of cases", ylab = "Cumulative", main = "Lift Chart", type = "l")
lines(c(0, sum(valid.data$Spam)) ~ c(0, nrow(valid.data)), lty = 2)

#glm.fwd decile chart
gain2fwd <- gains(valid.data$Spam, pred.fwd.valid)
barplot(gain2fwd$mean.resp / mean(df$actual), names.arg = gain2fwd$depth, xlab = "Percentile", 
        ylab = "Mean Response", main = "Decile-wise Lift Chart")


##reduced model using stepwise regression

#stepwise regression
glm.step <- step(glm.null, scope = list(glm.null, upper = logit.reg), direction = "both")
summary(glm.step)
#stepwise model deviance, AIC, BIC
glm.step$deviance
AIC(glm.step)
BIC(glm.step)

#glm.step out of sample prediction
pred.step.valid <- predict(glm.step, valid.data)

#glm.step confusionmatrix
confusionMatrix(as.factor(ifelse(pred.step.valid >= 0.5, "1", "0")),
                as.factor(valid.data$Spam), positive = "1")
#glm.step lift chart
gainstep <- gains(valid.data$Spam, pred.step.valid, groups = length(pred.step.valid))
plot(c(0, gainstep$cume.pct.of.total * sum(valid.data$Spam)) ~ c(0, gainstep$cume.obs),
     xlab = "# of cases", ylab = "Cumulative", main = "Lift Chart", type = "l")
lines(c(0, sum(valid.data$Spam)) ~ c(0, nrow(valid.data)), lty = 2)

#glm.step decile chart
gain2step <- gains(valid.data$Spam, pred.step.valid)
barplot(gain2step$mean.resp / mean(df$actual), names.arg = gain2step$depth, xlab = "Percentile", 
        ylab = "Mean Response", main = "Decile-wise Lift Chart")




