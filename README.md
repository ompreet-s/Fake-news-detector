# 🔍 FakeScope — Fake News Detector

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-black?style=flat&logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange?style=flat&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

A machine learning web application that detects **fake news** in real time using a **Gradient Boosting** classifier trained on 44,000+ news articles.

---

## 🌐 Live Demo
> Coming soon — deploy on Render for a live link

---

## 📸 Screenshot

> Add a screenshot of your website here after deployment

---

## 🧠 How It Works

1. User pastes a news headline or article
2. Text is cleaned and preprocessed
3. TF-IDF converts text into numerical features
4. Gradient Boosting model predicts REAL or FAKE
5. Confidence score and probabilities are displayed

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Machine Learning | Scikit-learn (Gradient Boosting) |
| Text Features | TF-IDF Vectorizer |
| Backend | Python, Flask, Flask-CORS |
| Frontend | HTML, CSS, JavaScript |
| Dataset | Fake News Dataset (44K+ articles) |

## 📁 Project Structure

Fake-news-detector/
├── app.py                  ← Flask backend API
├── requirements.txt        ← Python dependencies
├── README.md               ← Project documentation
├── .gitignore              ← Files to ignore
├── fake_news_model.ipynb   ← Jupyter training notebook
└── templates/
└── index.html          ← Frontend website

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/ompreet-s/Fake-news-detector.git
cd Fake-news-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model and save it
Open `fake_news_model.ipynb` in Jupyter Notebook and run all cells.
This will generate `model.pkl` and `label_encoder.pkl` in your project folder.

### 4. Start the Flask server
```bash
python app.py
```

### 5. Open in browser

http://localhost:5000

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~98% |
| Algorithm | Gradient Boosting |
| Features | TF-IDF (10,000 features, bigrams) |
| Training Data | 44,000+ articles |
| Classes | REAL / FAKE |

---

## 📝 API Endpoint

### `POST /predict`

**Request:**
```json
{
  "text": "Your news article or headline here"
}
```

**Response:**
```json
{
  "verdict": "FAKE",
  "label": "FAKE",
  "confidence": 94.3,
  "real_prob": 5.7,
  "fake_prob": 94.3
}
```

---

## 📦 Requirements
flask
flask-cors
scikit-learn
xgboost
pandas
numpy

Install all with:
```bash
pip install -r requirements.txt
```

---

## ☁️ Deploy on Render (Free)

1. Push your code to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repository
4. Set the following:

| Setting | Value |
|---|---|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python app.py` |
| Environment | Python 3 |

5. Click **Deploy** — get a free public URL!

---

## ⚠️ Note on Model Files

The `model.pkl` and `label_encoder.pkl` files are **not included** in this repository because they exceed GitHub's file size limit. To generate them locally, simply run all cells in `fake_news_model.ipynb`.

---

## 👨‍💻 Author

**Ompreet**
- GitHub: [@ompreet-s](https://github.com/ompreet-s)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use and modify it.

---

## 🙏 Acknowledgements

- Dataset sourced from Kaggle Fake News dataset
- Built as part of AI Internship project
- Powered by Scikit-learn and Flask

---
How to add it to your repo
In your terminal:
# Create the README
# (paste the above content into README.md in your project folder)

git add README.md
git commit -m "Add README"
git push

## 📁 Project Structure
