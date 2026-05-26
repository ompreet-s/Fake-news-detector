from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import re
import string

app = Flask(__name__)
CORS(app)

# ── Load model and encoder ──
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# ── Same clean_text function from notebook ──
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Routes ──
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text or len(text) < 20:
            return jsonify({'error': 'Text too short'}), 400

        # Clean and predict
        cleaned = clean_text(text)
        pred = model.predict([cleaned])[0]
        proba = model.predict_proba([cleaned])[0]

        label = le.inverse_transform([pred])[0]   # 'REAL' or 'FAKE'
        confidence = round(float(max(proba)) * 100, 1)

        # Determine verdict
        if confidence < 60:
            verdict = 'UNCERTAIN'
        else:
            verdict = label

        return jsonify({
            'verdict': verdict,
            'label': label,
            'confidence': confidence,
            'real_prob': round(float(proba[1]) * 100, 1),
            'fake_prob': round(float(proba[0]) * 100, 1)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)