# Repo for our CMPE 255 Data Mining project

## Steps to run the app:

1. Clone the repo

2. Run "pip install -r requirements.txt"

3. Run "python app.py"

## Objectives:
- Stream Reddit data
- Using NLP classify the subreddit post from each Reddit as positive or negative
- Use the KDD (Knowledge discovery from data) approach

!["Image of approach and methodology"](https://github.com/varun-bhaseen/Reddit-Classification-Project/blob/master/images/DM%20Project%20Report.jpg)

## Web Deployment Technologies used:
- Database Streaming: Apache Spark
- Infrastructure- AWS cloud
- Application Web Framework: Python-Flask
- API: Reddit API
- Notebook: Jupyter

## Machine Learning Model Building:
- Data gathering: Reddit API
- Data Preprocessing: NLTK (for stemming, lemmatization, stop words removal, Tokenization)
- Feature Engineering: TF-IDF vectorization
- Feature Engineering: Count Vectorization
- Feature Extraction: Vader (Sentiment analysis and word polarity detection)
- Visualization: Word Cloud, Matplot Lib, Seaborn
- Model Building and Tuning: (SciKit Learn) Multinomial Naive Bayes (AUC: 85.82% ); Logistic Regression (AUC: 83.24%); SVC (AUC: 93.63%)
- Model saved as Pickle
