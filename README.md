# Emergency-Disaster-Message-Classification
## 1. Overview
Figure eight has provided labelled data of messages that were sent during various disasters. In this project I have created a machine learning pipeline that would classify these messages into different categories. There are 3 key aspects of this project-
1) **ETL pipeline** : Cleaning the data and sending it into a database by developing an ETL pipeline.
2) **Machine Learning Pipeline** : Running different algorithms and finding the best F1 score on the test data. Also using GridSearch to find the optimal hyper-parameters.
3) **Flask app**- Developing a web app that displays a summary of the training data and classifies new messages.

F1 score was the key metric that I was using to analyze the performance of the model. <br>
Random Forest Classifier helped me get a test F1 score of **0.65** when I micro averaged it across the multi-label output. <br>
Ada Boost Classifier helped to get a test F1 score of **0.67** on micro averaging it across multi-label output. This classifier was used in the web app.

## 2. Files Structure
All the files are divided into following parts-
#### a) App folder:
It contains the following files-
- **run.py** file defines the app route. It also contains the plotly visualizations present in the webapp.
- templates folder contains **go.html** and **master.html** files. 
    - **Master.html** reders the emergency message classifier.
    - **go.html** renders the visualizations made from plotly.

#### b) Data Folder:
It contains the following files-
- **Categories.csv** file contains the categories of the messages i.e. the labelled output.
- **Messages.csv** file contains the raw text messages.
- **Disaster.db** is a database in which the merged csv files are stored after cleaning.
- **process_data.py** is the ETL script that includes all the loading and cleaning steps.

#### c) Models Folder:
It contains the following files-
- **train_classifier.pkl** file contains the machine learning pipeline that includes the tokenization of text and use of random forest and Ada boost classifier. 
- **classifier.pkl** file contains the best model that was found after running the GridSearch. Since Ada-Boost classifier was found to be performing better at the test F1 score (0.67) it was saved in the pickle file.

There are also the ETL and Machine learning Jupyter notebooks present that show the preliminary analysis.

## 3. Installation
Pandas, Numpy, SQLAlchemy, sklearn and pickle were the key libraries used in this project. However, in order to build an NLP pipeline NLTK package was extensively used.
Corpus, Tokenize and Stem were the modules mainly used from the NLTK package.

To rerun the scripts follow the instructions below-
a) Clone this repo to your desktop.
b) Go to the root directory that has the data and the model from your terminal.
c) To run the ETL script type: *python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db*
d) To run the Machine Learning pipeline scipt type: *python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl*
e) To run the web app type: *python app/run.py*

