# Naive Bayes

# Importing the dataset
dataset = read.csv('nurseryNVB.csv', 
                   col.names = c('parents','has_nurs','form','children','housing','finance','social','health','class'), 
                   header = FALSE)

# Encoding the target feature as factor
dataset$class = factor(dataset$class, levels = c('not_recom' ,'priority', 'recommend', 'spec_prior', 'very_recom'),
                       labels = c(0,1,2,3,4))
dataset$parents = factor(dataset$parents, levels = c('great_pret' ,'pretentious', 'usual'),
                       labels = c(0,1,2))
dataset$has_nurs = factor(dataset$has_nurs, levels = c('proper','less_proper', 'improper' ,   'critical'  ,  'very_crit'),
                         labels = c(0,1,2,3,4))
dataset$form = factor(dataset$form, levels = c('complete', 'completed', 'foster', 'incomplete'),
                          labels = c(0,1,2,3))
dataset$children = factor(dataset$children, levels = c('1' ,'2', '3', 'more'),
                          labels = c(0,1,2,3))
dataset$housing = factor(dataset$housing, levels = c('convenient', 'critical' ,'less_conv'),
                          labels = c(0,1,2))
dataset$finance = factor(dataset$finance, levels = c('convenient', 'inconv'),
                         labels = c(0,1))
dataset$social = factor(dataset$social, levels = c('nonprob', 'problematic', 'slightly_prob'),
                         labels = c(0,1,2))
dataset$health = factor(dataset$health, levels = c('not_recom', 'priority', 'recommended'),
                         labels = c(0,1,2))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$class, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Naive Bayes to the Training set
# install.packages('e1071')
library(e1071)
classifier = naiveBayes(x = training_set[-9],
                        y = training_set$class)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-9])

# Making the Confusion Matrix
cm = table(test_set[, 9], y_pred)
#      0    1    2    3    4
# 0 1080    0    0    0    0
# 1    0  967    0   99    0
# 2    0    0    0    0    0
# 3    0  139    0  872    0
# 4    0   76    0    0    6

# Accuracy of the model
ac = sum(diag(cm))/sum(cm)
# ac =  0.9030565 i.e.90.30