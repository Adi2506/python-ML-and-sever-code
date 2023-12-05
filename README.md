# python-ML-and-sever-code
This raspatory contains code for SVM model and Bernoulli Naive Bayes which are used for sentiment analysis and spam detection and used saved with python library joblib and integrated with app developed in flutter using Flask API it is easy simple project and this raspatory also contain Flask api code 

spamModel.py:

This file contains code for brnoulli naive bayes model which is trained for comments spam analysis downloaded from webbsites like kaggle I am not providding the dataset and by using joblib library model is saved so while implements it doesn't train again and again.It classifies and visulizes comments in spam and non spam It have 80-90 percent accuracy.

spamServer.py :

It is used in initial steps to make server using Flask api so it canintegrate with flutter application

svm.py : It is same as spamModel just for sentiment analysis do same task diffrence is classification task is sentiment analysis comments are divided into positive negative and neutral it is trained on dataset downloaded from web I am not providing dataset

example: it is same server making code for svm classifier

In both servers by using joblib library models are loaded as server need to run multiple times so if whole svm and spam code is used then model will train every time server is operating so here joblib libarary is used Flutter code for the same project done by another team mate which I doesn't have in my machine I will try to provide it as soon as possible

While implementing it with Flutter we integrated code of example.py and spamServer.py I will try to provide it as soom as possible
