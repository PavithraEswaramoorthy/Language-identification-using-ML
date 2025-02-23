# Language Detection Model

## Overview
This project focuses on building a machine learning model to detect the language of a given text snippet. The model is trained on a dataset containing text samples from 17 languages. The implemented algorithms include **Multinomial Naive Bayes** and **Decision Tree**, with the former achieving significantly higher accuracy.

## Dataset
- **Filename**: `Language Detection.csv`
- **Columns**: `Text` (text samples), `Language` (language labels)
- **Languages**: 17 languages including English, French, Spanish, Kannada, Hindi, and more.
- **Rows**: 10,337 text samples.

## Key Steps
1. **Data Loading & Preprocessing**
   - Clean text by removing special characters/digits and converting to lowercase.
   - Encode language labels using `LabelEncoder`.
   
2. **Feature Extraction**
   - Convert text to numerical features using `CountVectorizer`.

3. **Model Training**
   - **Multinomial Naive Bayes**: Achieved **97.7% accuracy**.
   - **Decision Tree**: Achieved **85% accuracy** (for comparison).

4. **Evaluation**
   - Accuracy, confusion matrix, and classification report.
   - Visualization of confusion matrix using `seaborn`.

5. **Prediction Function**
   - `predict(text)` function to detect the language of new text inputs.

## Dependencies
- Python 3.x
- Libraries: 
  ```bash
  pandas, numpy, matplotlib, seaborn, scikit-learn, re
  ```

## Usage
1. **Load Data**:
   ```python
   df = pd.read_csv('/content/Language Detection.csv')
   ```
2. **Preprocess Data**:
   - Clean text and encode labels.
3. **Train Model**:
   ```python
   model = MultinomialNB()
   model.fit(x_train, y_train)
   ```
4. **Evaluate**:
   - Accuracy score, confusion matrix, and classification report.
5. **Predict**:
   ```python
   predict("Enter your text here")  # Output: The language is in [Detected_Language]
   ```

## Results
- **Classification Report** (Naive Bayes):
  ```
              precision    recall  f1-score   support
  English       1.00      0.97      0.99       102
  French        0.99      0.92      0.96        92
  ... (17 languages)
  Accuracy: 97.7%
  ```
