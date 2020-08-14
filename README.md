# Emergency-Disaster-Message-Classification
## 1. Overview
Figure eight has provided labelled data of messages that were sent during various disasters. In this project I have created a machine learning pipeline that would classify these messages into different categories. There are 3 key aspects of this project-
1) **ETL pipeline** : Cleaning the data and sending it into a database by developing an ETL pipeline.
2) **Machine Learning Pipeline** : Running different algorithms and finding the best F1 score on the test data. Also using GridSearch to find the optimal hyper-parameters.
3) **Flask app**- Developing a web app that displays a summary of the training data and classifies new messages.

F1 score was the key metric that I was using to analyze the performance of the model. 
Random Forest Classifier helped me get a test F1 score of **0.65** when I micro averaged it across the multi-label output.
Support Vector Machine Classifier helped to get a test F1 score of ** ** on micro averaging it across multi-label output.

## 2. Installation
Pandas, Numpy, SQLAlchemy, sklearn and pickle were the key libraries used in this project. However, in order to build an NLP pipeline NLTK package was extensively used.
Corpus, Tokenize and Stem were mainly used from the NLTK package.
