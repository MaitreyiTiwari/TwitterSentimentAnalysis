# Twitter Sentiment Analysis

## Project Overview

This project implements a machine learning solution to classify Twitter sentiment as positive or negative. The model is trained on a large dataset of 1.6 million tweets and uses Natural Language Processing (NLP) techniques combined with Logistic Regression to predict sentiment polarity.

## Dataset

The project utilizes the **Sentiment140** dataset from Kaggle, which contains tweets labeled with sentiment scores.

### Dataset Details
- **Source**: Kazanova's Sentiment140 dataset (Kaggle)
- **Total Tweets**: 1,600,000
- **File**: training.1600000.processed.noemoticon.csv
- **Encoding**: ISO-8859-1
- **File Size**: Approximately 400+ MB

### Data Structure
The dataset contains 6 columns:
1. **target**: Sentiment label (0 = Negative, 4 = Positive)
2. **ids**: Tweet ID
3. **date**: Tweet timestamp
4. **flag**: Query flag
5. **user**: Username
6. **text**: Tweet content

### Data Statistics
- **Total Records**: 1,600,000 tweets
- **Class Distribution**: Balanced dataset with 800,000 positive and 800,000 negative tweets
- **Missing Values**: None (dataset is complete)
- **Target Variable Labels**: 0 (Negative), 1 (Positive) - converted from original [0, 4] to [0, 1]

## Project Architecture

### 1. Data Loading and Preparation
- Dataset is downloaded from Kaggle using the Kaggle API
- CSV file is extracted and loaded using pandas with proper encoding handling
- Data validation performed to check for null values

### 2. Data Preprocessing
The project implements a comprehensive text preprocessing pipeline using the `stemming()` function:

#### Preprocessing Steps:
1. **Remove Special Characters**: All non-alphabetic characters (numbers, punctuation, emojis) are replaced with spaces using regex pattern `[^a-zA-Z]`
2. **Lowercase Conversion**: Text is converted to lowercase for uniformity
3. **Tokenization**: Text is split into individual words
4. **Stopword Removal**: Common English stopwords are removed using NLTK's English stopword corpus
5. **Stemming**: Words are reduced to their root form using Porter Stemmer algorithm

**Purpose**: These preprocessing steps normalize the text data, reduce feature dimensionality, and focus on meaningful content while removing noise.

### 3. Feature Engineering
- **TF-IDF Vectorization**: Text data is converted into numerical features using TfidfVectorizer
- This approach measures word importance relative to the entire corpus
- Creates sparse matrix representation suitable for machine learning models

### 4. Model Training and Evaluation

#### Model: Logistic Regression
- **Algorithm**: Binary Logistic Regression
- **Maximum Iterations**: 1000 (to ensure convergence)
- **Activation Function**: Sigmoid (inherent to logistic regression)

#### Train-Test Split
- **Training Set**: 80% of data (1,280,000 tweets)
- **Testing Set**: 20% of data (320,000 tweets)
- **Stratification**: Ensures balanced class distribution in both sets
- **Random State**: 2 (for reproducibility)

#### Model Performance
- **Training Accuracy**: Approximately 78-80%
- **Testing Accuracy**: Approximately 77-79%
- The close training and testing accuracy indicates the model generalizes well without significant overfitting

## Key Libraries and Dependencies

### Data Processing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations

### Natural Language Processing
- **nltk**: Stopword corpus and text processing utilities
- **PorterStemmer**: Stemming algorithm for word normalization

### Machine Learning
- **scikit-learn**:
  - TfidfVectorizer: Feature extraction
  - LogisticRegression: Classification model
  - train_test_split: Dataset splitting
  - GridSearchCV: Hyperparameter tuning (imported for potential use)
  - accuracy_score: Model evaluation metric
  - classification_report: Detailed performance metrics
  - confusion_matrix: True/False positive/negative analysis
  - Pipeline: Model pipeline construction

### Utilities
- **zipfile**: Dataset extraction
- **pickle**: Model serialization and persistence
- **re**: Regular expression operations for text cleaning

## Model Persistence

The trained model is saved using Python's pickle library:
- **File**: sentiment_model.pkl
- **Format**: Binary pickle format
- **Purpose**: Enables model reuse without retraining

## Workflow Summary

1. Import required libraries and set up environment
2. Download Sentiment140 dataset from Kaggle
3. Extract and load the dataset
4. Perform exploratory data analysis (shape, null checks, class distribution)
5. Normalize target variable (convert 4 to 1 for binary classification)
6. Apply preprocessing pipeline (text cleaning, stemming, stopword removal)
7. Prepare features (X) and labels (y)
8. Split data into training (80%) and testing (20%) sets with stratification
9. Vectorize text using TF-IDF
10. Train Logistic Regression model on training set
11. Evaluate performance on both training and testing sets
12. Save trained model to disk

## Performance Insights

- The model achieves approximately 78-79% accuracy on both training and testing sets
- This performance level is respectable for binary sentiment classification on Twitter data
- The balanced accuracy between training and testing sets indicates good generalization
- Potential for improvement exists through hyperparameter tuning, alternative algorithms, or ensemble methods


## Usage

To use this project:

1. Ensure all dependencies are installed:
   ```bash
   pip install pandas numpy nltk scikit-learn

2. Download NLTK stopwords (if not already downloaded):
  import nltk
  nltk.download('stopwords')
  
3. Set up Kaggle API credentials for dataset download
4. Run the notebook cells sequentially to:
 -Download and extract the dataset
 -Train the model
 -Evaluate performance

5. Load the saved model for predictions:
import pickle
model = pickle.load(open('sentiment_model.pkl', 'rb'))




