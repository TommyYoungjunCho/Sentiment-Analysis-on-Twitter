# Sentiment-Analysis-on-Twitter
**Project Overview**

This project aims to classify and detect the degree of positivity in Twitter texts using supervised machine learning techniques. The sentiment analysis classifies tweets into three sentiments: negative, neutral, and positive. The project includes data preprocessing, feature selection, model training, and evaluation. The goal is to achieve high accuracy in sentiment classification to provide valuable insights from social media data.

**Technologies Used**

- **Programming Languages**: Python
- **Libraries**: NLTK, Scikit-learn, XGBoost, Pandas, NumPy
- **Tools**: Jupyter Notebook, GitHub

**Project Achievements**

- **High-Quality Preprocessing**:
    - Effectively preprocessed text data, including emoji conversion, tokenization, and lemmatization, to improve model performance.
- **Comprehensive Model Evaluation**:
    - Evaluated multiple machine learning models and identified the Naïve Bayes model as the best performer for sentiment analysis.
- **Insightful Results**:
    - Provided valuable insights into the distribution of sentiments in Twitter data, highlighting the challenges of imbalanced datasets.

**Period**

- 2022.3 ~ 2022.6

**GitHub Repository**

- https://github.com/TommyYoungjunCho/Sentiment-Analysis-on-Twitter

# Project Details

---

1. **Data Description**:
    - **Dataset**: Tweets collected from Twitter with labeled sentiments (negative, neutral, positive).
2. **Data Exploration and Preprocessing**:
    - **Emoji Conversion**: Replaced emojis with corresponding words based on their meanings.
    - **Tokenization**: Divided text into tokens using whitespace and punctuation-based tokenization.
    - **Stopwords Removal**: Removed meaningless words, URLs, HTML tags, digits, hashtags, and mentions.
    - **Part-of-Speech Tagging**: Tagged parts of speech and retained only verbs, adjectives, nouns, and adverbs.
    - **Lemmatization**: Reduced words to their root forms to minimize complexity.
    
3. **Feature Engineering**:
    - **Bag of Words (BoW)**: Created numerical vectors by counting word occurrences.
    - **Term Frequency-Inverse Document Frequency (TF-IDF)**: Calculated the importance of words based on their frequency in documents.
    
4. **Machine Learning Models**:
    - **Decision Tree (DT)**: Used to create a model that predicts target variables by learning decision rules.
    - **Multinomial Naïve Bayes (NB)**: Probabilistic model based on Bayes' theorem for classification.
    - **Support Vector Machine (SVM)**: Mapped data to high-dimensional feature space to find a separator between categories.
    - **Extreme Gradient Boosting (XGB)**: Optimized gradient boosting library for efficient and accurate predictions.
    
5. **Evaluation Metrics**:
    - **Confusion Matrix**: Compared predicted values to actual values to measure prediction performance.
    - **Accuracy**: Measured the proportion of correctly predicted instances.
    - **F1 Score**: Considered precision and recall for performance evaluation.

1. **Results and Discussion**:
    - **Data Distribution**: Found that 58% of the data was neutral, 25% positive, and 17% negative, which could skew model predictions.
    - **Model Performance**:
        - **Decision Tree (DT)**: Achieved 59.6% training accuracy and 37.4% test accuracy.
        - **Naïve Bayes (NB)**: Highest performance with 84.8% training accuracy and 55.1% test accuracy. Best accuracy of 58.7% with BoW preprocessing.
        - **Support Vector Machine (SVM)**: Achieved 75.5% training accuracy and 54.3% test accuracy.
        - **Extreme Gradient Boosting (XGB)**: Achieved 82.6% training accuracy and 55% test accuracy


## Notion Portfolio Page
- [[Notion Portfolio Page Link](https://magic-taleggio-e52.notion.site/Portfolio-705d90d52e4e451488fb20e3d6653d3b)](#) 
