# 💬 Sentiment Analysis Web Application

This project is a **Machine Learning-based web application** that performs sentiment analysis on user-provided text (such as tweets, reviews, or comments) and classifies it as **Positive 😊** or **Negative 😠**.

The application is built using **Natural Language Processing (NLP)** techniques and deployed through an interactive **Streamlit web interface**.

---

## 🚀 Features

* ✅ Real-time sentiment prediction from user input
* ✅ Machine Learning model (Logistic Regression)
* ✅ TF-IDF vectorization for text feature extraction
* ✅ Clean and modular code structure
* ✅ Interactive UI using Streamlit
* ✅ Fast and efficient prediction system

---

## 🧠 Technologies Used

* **Python**
* **Scikit-learn** (Machine Learning)
* **Pandas & NumPy** (Data Processing)
* **NLTK** (Text preprocessing)
* **Streamlit** (Web application framework)
* **Joblib** (Model saving/loading)

---

## 📁 Project Structure

```
sentiment-analysis-app/
│
├── app.py                # Streamlit application (frontend + backend)
├── train.py              # Model training script
├── utils.py              # Text preprocessing functions
├── twitter_fetch.py      # Placeholder for API-based tweet fetching
├── model.pkl             # Trained machine learning model
├── vectorizer.pkl        # TF-IDF vectorizer
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

---

## ⚙️ How It Works

1. The user enters a sentence or text input
2. The input text is cleaned and preprocessed
3. Text is converted into numerical features using TF-IDF
4. The trained model predicts sentiment
5. Output is displayed as **Positive 😊** or **Negative 😠**

---

## ▶️ How to Run the Project Locally

### 🔹 Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🔹 Step 2: Train the Model (Run Once)

```bash
python train.py
```

---

### 🔹 Step 3: Launch the Application

```bash
streamlit run app.py
```

---

## 📊 Example

**Input:**

```
I love this product!
```

**Output:**

```
Positive 😊
```

---

## 🎯 Project Objective

The main objective of this project is to:

* Understand Natural Language Processing (NLP)
* Apply Machine Learning techniques for text classification
* Build an end-to-end AI application
* Create a real-world deployable web application

---

## 🚀 Future Enhancements

This project can be further improved by adding:

* 🔥 Deep Learning models (BERT, LSTM)
* 🔥 Real-time social media data integration
* 🔥 Dashboard analytics and visualization
* 🔥 Multi-language sentiment analysis
* 🔥 Voice input support
* 🔥 CSV bulk sentiment analysis

---

## 🧠 Key Learnings

* Text preprocessing and cleaning techniques
* Feature extraction using TF-IDF
* Model training and evaluation
* Building interactive applications with Streamlit
* Structuring real-world ML projects

---

## 👨‍💻 Author

**Mahesh Dattatreya**
Final Year B.Tech – Artificial Intelligence & Data Science

---

## ⭐ Support

If you found this project helpful:

⭐ Star this repository on GitHub
📌 Share it with others

---

## 📌 Note

This project is built for **educational and learning purposes** and demonstrates the practical implementation of Machine Learning and NLP concepts in a real-world scenario.
