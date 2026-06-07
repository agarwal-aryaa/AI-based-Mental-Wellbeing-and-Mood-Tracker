# AI-based-Mental-Wellbeing-and-Mood-Tracker
AI-powered mood journaling app using Flask, TF-IDF, and ML models for emotion detection, mood tracking, analytics, and wellness recommendations.

## Overview

Mental health issues such as depression and anxiety affect millions of people worldwide. Most mood-tracking applications rely on manual user input and lack intelligent analysis or early warning systems.

SmartJournal addresses this challenge by:

* Automatically classifying journal entries into meaningful mood categories.
* Detecting long-term negative emotional patterns.
* Providing personalized wellness recommendations.
* Offering real-time AI-powered mood monitoring.

## Features

### Core Features

* Live AI Mood Prediction – Instantly predicts the user's mood while typing.
* AI Chatbot – Provides context-aware responses and wellness suggestions.
* Mood Trend Dashboard – Displays mood history and statistics through interactive charts.
* Mood Streak Detection – Detects continuous negative mood patterns and generates alerts.
* History Log – Maintains a record of journal entries, predictions, confidence scores, and timestamps.
* Guardian Alerts – Supports emergency contact information for critical situations.
* User Authentication – Secure registration and login system.
* ML Analysis Dashboard – Displays confusion matrices, model metrics, and performance comparisons.

## AI/ML Concepts Used

### Text Preprocessing

* URL removal
* Mention removal
* Text normalization
* Lowercasing
* Noise reduction

### Feature Engineering

* TF-IDF Vectorization
* Unigrams and Bigrams
* Maximum 40,000 features

### Machine Learning Models

* Linear SVM
* Logistic Regression
* Decision Tree
* k-Nearest Neighbors (k-NN)

### Mood Categories

| Mood Category | Mapped Emotions                                    |
| ------------- | -------------------------------------------------- |
| Happy         | happiness, love, enthusiasm, fun, relief, surprise |
| Neutral       | neutral, boredom, empty                            |
| Sad           | sadness                                            |
| Anxious       | worry                                              |
| Angry         | hate, anger                                        |

## Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Linear SVM          | ~46.5%   |
| Logistic Regression | ~44%     |
| k-NN                | ~39%     |
| Decision Tree       | ~34%     |

Linear SVM achieved the highest accuracy and is used as the primary classification model.

## Technology Stack

### Backend

* Python 3.x
* Flask 2.3+

### Machine Learning

* Scikit-learn
* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn
* Chart.js

### Frontend

* HTML5
* CSS3
* JavaScript

### Storage

* JSON

### Dataset

* tweet_emotions.csv
* Approximately 40,000 labelled tweets

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/SmartJournal.git
cd SmartJournal
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python train_model.py
```

### Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```
## Contributors

- Vanshika Mahindru
- Aryaa Agarwal  
- Ratna Srivastava  


