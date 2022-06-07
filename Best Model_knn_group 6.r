#Load the data
load(file = "class_data.RData")
#converting dependent variable to a categorical format since it's two classes: 0 & 1
y<-as.factor(y)
set.seed(22)

#Storing scaled version of x in a separate Data frame for future use (Ridge, Lasso, PCA, KNN)
scl_x = scale(x)

##KNN with original data
library(class)
library(Boruta)
library(pROC)
library(ROCit)
library(ROCR)

knn_error = data.frame()
knn_auc = data.frame()
K=5

#K fold cross validation loop 
for(i in 1:K){
  
  #Splitting data into training and validation sets
  train__ <- scl_x[Outer_folds!=i,]
  train_y <- y[Outer_folds!=i]
  
  test__ <- scl_x[Outer_folds==i,]
  test_y <- y[Outer_folds==i]
  
  #Utilizing Boruta algorithm on training data() for feature selection
  boruta_output <- Boruta(train__,train_y)
  
  #Print significant variables including tentative
  boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
  
  #Only using selected features for knn
  train_x<-train__[,boruta_signif]
  test_x<-test__[,boruta_signif]
  
  #Iterating through k nearest neighbours for each fold
  neighbours=dim(train_x)[1]
  for(j in 1:neighbours){
    knnmodel = knn(train=train_x, cl=train_y ,test=test_x, k=j, prob =TRUE)
    knn_error[i,j]=mean(knnmodel!=test_y)
    knn_auc[i,j] <- roc(test_y,attributes(knnmodel)$prob)$auc
  }
}
#storing mean cv error and auc score for each value of k
mean_fold_error = apply(knn_error,2,mean)
knn_which.neighbour.min=which.min(mean_fold_error)

mean_fold_auc = apply(knn_auc,2,mean)
auc_k =mean_fold_auc[knn_which.neighbour.min]

#Printing lowest cv error and highest auc score where no of neighbours is equal to the variable number displayed in name
print(mean_fold_error[knn_which.neighbour.min])
print(auc_k)

test_error <-mean_fold_error[knn_which.neighbour.min]

#Variation of cv error with no of neighbours
plot(1:320,mean_fold_error,xlab ="No of Neighbours", ylab = "Mean CV Error")

#Utilizing Boruta algorithm on training data() for feature selection
boruta_output2 <- Boruta(scl_x,y)

#Significant variables including tentative
boruta_signif2 <- getSelectedAttributes(boruta_output2, withTentative = TRUE)

#Creating final model and making prediction based on xnew
xfinal_train<-scl_x[,boruta_signif2]
xfinal_test<-scale(xnew)[,boruta_signif2]

ynew = knn(train=xfinal_train, cl=y ,test=xfinal_test, k=6)
save(ynew, test_error,file="6.RData")