
The major goal of this project is to predict financial re-
cession given the frequencies of the top 500 word stems in the reports of
financial companies. After applying various learning models, we can see that
the prediction of financial recession by the bag of words has an accuracy of
more than 90%. Hence, there is indeed a correlation between the two.
Moreover, we have compared different learning models (ensemble methods
with Decision Tree, SVM, and KNN) with various parameters to find the best
model with a relatively high average accuracy and low variance of accuracy
by cross-validation on the training data set. In addition, we have also tried
several pre-processing methods (tf-idf, feature selection, and centroid-based
clustering) to improve the accuracy of the learning models. In the end, the
best model is Gradient Boosting with Decision Tree using the pre-processed
tf-idf data set.
