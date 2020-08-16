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
It contains the following files- <br>
- **run.py** file defines the app route. It also contains the plotly visualizations present in the webapp.  <br>
- templates folder contains **go.html** and **master.html** files. <br>
    - **Master.html** reders the emergency message classifier.<br>
    - **go.html** renders the visualizations made from plotly.<br>

#### b) Data Folder:
It contains the following files-<br>
- **Categories.csv** file contains the categories of the messages i.e. the labelled output.<br>
- **Messages.csv** file contains the raw text messages.<br>
- **Disaster.db** is a database in which the merged csv files are stored after cleaning.<br>
- **process_data.py** is the ETL script that includes all the loading and cleaning steps.<br>

#### c) Models Folder:
It contains the following files-<br>
- **train_classifier.pkl** file contains the machine learning pipeline that includes the tokenization of text and use of random forest and Ada boost classifier. <br>
- **classifier.pkl** file contains the best model that was found after running the GridSearch. Since Ada-Boost classifier was found to be performing better at the test F1 score (0.67) it was saved in the pickle file.<br>

There are also the ETL and Machine learning Jupyter notebooks present that show the preliminary analysis.<br>


## 3. Installation
Pandas, Numpy, SQLAlchemy, sklearn and pickle were the key libraries used in this project. However, in order to build an NLP pipeline NLTK package was extensively used.
Corpus, Tokenize and Stem were the modules mainly used from the NLTK package.

To rerun the scripts follow the instructions below- <br>
a) Clone this repo to your desktop. <br>
b) Go to the root directory that has the data and the model from your terminal. <br>
c) To run the ETL script type: *python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db* <br>
d) To run the Machine Learning pipeline scipt type: *python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl* <br>
e) To run the web app type: *python app/run.py* <br>
f) Open localhost:3001 on the web browser to access the webapp on the local desktop. <br>

The screenshot of the web-app is attached as a file.


## 4. License
The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
