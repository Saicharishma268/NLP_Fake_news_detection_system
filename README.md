# 📰 Fake News Detection System Using NLP and Machine Learning

## 📌 Project Overview

The **Fake News Detection System** is an NLP-based machine learning project that identifies whether a given news article is **FAKE** or **GENUINE**.

With the rapid growth of digital media and social networking platforms, misleading information can spread quickly. This project uses Natural Language Processing (NLP) techniques and Machine Learning to analyze news content and classify it based on patterns learned from a labeled dataset.

The system uses **TF-IDF (Term Frequency–Inverse Document Frequency)** for text feature extraction and **Logistic Regression** for classification. A simple interactive dashboard is developed using **Gradio**, allowing users to enter a news article and view the predicted result.

## 🎯 Objectives

- To understand and implement Natural Language Processing techniques.
- To collect and preprocess fake and genuine news datasets.
- To convert textual data into numerical features using TF-IDF.
- To train a Logistic Regression classification model.
- To evaluate the performance of the trained model.
- To develop an interactive dashboard for news prediction.
- To classify user-provided news articles as FAKE or GENUINE.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Google Colab | Development environment |
| Pandas | Dataset loading and manipulation |
| NLTK | Text preprocessing |
| Scikit-learn | Machine learning and evaluation |
| TF-IDF Vectorizer | Text feature extraction |
| Logistic Regression | News classification |
| Gradio | Interactive web dashboard |

## 📂 Dataset Description

The project uses two CSV files:

1. **Fake.csv** – Contains fake news articles.
2. **True.csv** – Contains genuine news articles.

The datasets contain news-related information such as:

- Title
- Text
- Subject
- Date

### Dataset Labeling

A numerical label is assigned to each category:

| News Category | Label |
|---|---:|
| Fake News | 0 |
| Genuine News | 1 |

Both datasets are combined into a single dataset for further processing.

## 🔄 Project Workflow

The project follows these steps:

1. Dataset Collection
2. Data Loading
3. Data Understanding
4. Data Cleaning
5. Label Assignment
6. Dataset Combination
7. Content Creation
8. NLP Text Preprocessing
9. Train-Test Splitting
10. TF-IDF Feature Extraction
11. Logistic Regression Model Training
12. News Prediction
13. Model Evaluation
14. Gradio Dashboard Development

## 🧹 1. Data Preprocessing

Before training the model, the dataset is cleaned and prepared.

The preprocessing steps include:

- **Removing duplicates:** Eliminates repeated records.
- **Removing missing values:** Removes incomplete records.
- **Labeling:** Assigns 0 to fake news and 1 to genuine news.
- **Combining datasets:** Merges fake and genuine news records.
- **Content creation:** Combines the title and article text into one content column.

## 📝 2. NLP Text Preprocessing

Natural Language Processing techniques are applied to convert raw news text into a cleaner format.

The following operations are performed:

- **Lowercasing:** Converts all text into lowercase.
- **Removing unwanted characters:** Removes unnecessary symbols and special characters using regular expressions.
- **Tokenization:** Splits sentences into individual words.
- **Stopword Removal:** Removes commonly occurring words that provide limited useful information for this task.
- **Stemming:** Reduces words to their root forms using the Porter Stemming algorithm.

These steps help prepare the text for feature extraction.

## 📊 3. Train-Test Splitting

The cleaned dataset is divided into two parts:

- **Training Data – 80%:** Used to train the machine learning model.
- **Testing Data – 20%:** Used to evaluate the trained model on unseen records.

The `train_test_split()` function from Scikit-learn is used with `random_state=42` and `stratify=y`.

Stratification helps maintain the class proportions in both training and testing datasets.

## 🔢 4. TF-IDF Feature Extraction

TF-IDF stands for **Term Frequency–Inverse Document Frequency**.

It is a numerical feature extraction technique that converts text into numerical values so that machine learning algorithms can process it.

### Term Frequency (TF)

Measures how frequently a particular word appears in a document.

### Inverse Document Frequency (IDF)

Measures how important or informative a word is across the collection of documents. Words appearing in many documents generally receive lower IDF values.

### TF-IDF Formula

TF-IDF is calculated using:

**TF-IDF = TF × IDF**

### Implementation

The project uses:

`TfidfVectorizer(max_features=5000)`

This selects a vocabulary of up to 5,000 features based on the training data.

- `fit_transform(X_train)` learns the vocabulary and IDF values from the training data and converts the training text into a TF-IDF matrix.
- `transform(X_test)` uses the same learned vocabulary and IDF values to convert the testing data.

Using the training vocabulary and IDF values for the testing data helps prevent data leakage.

## 🤖 5. Machine Learning Model – Logistic Regression

Logistic Regression is a supervised machine learning classification algorithm.

It learns patterns from labeled training data and predicts the category of new input data.

In this project:

- Input: TF-IDF numerical features of news articles.
- Output: Fake News (0) or Genuine News (1).

The Logistic Regression model is trained using the TF-IDF features and their corresponding labels.

## 📈 6. Model Evaluation

After training, the model is evaluated using the testing dataset.

The evaluation includes:

- **Accuracy:** Measures the proportion of correctly classified news articles.
- **Confusion Matrix:** Shows correct and incorrect predictions for both classes.
- **Precision:** Measures how many articles predicted as a particular class actually belong to that class.
- **Recall:** Measures how many actual articles of a particular class are correctly identified.
- **F1-Score:** Represents the harmonic mean of precision and recall.

The project achieved approximately **98.65% accuracy** in the earlier reported test run. Actual results may vary if the dataset or preprocessing changes.

## 🖥️ 7. Gradio Dashboard

Gradio is used to develop an interactive interface for the project.

### Dashboard Functionality

1. The user enters a news article in the input textbox.
2. The article is passed to the prediction function.
3. The same text preprocessing steps are applied.
4. The trained TF-IDF vectorizer converts the article into numerical features.
5. The trained Logistic Regression model predicts the category.
6. The dashboard displays the result.

### Possible Outputs

- **FAKE NEWS**
- **GENUINE NEWS**

The dashboard makes the model easy to use without requiring users to run individual prediction commands.

## 📁 Project Structure

```text
NLP_Fake_news_detection_system/
│
├── Fake.csv
├── True.csv
├── combined_dataset.csv
│
├── nlp_project_progress.ipynb
│
├── README.md
│
└── screenshots/
   
