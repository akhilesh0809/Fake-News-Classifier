📰 Fake News Classifier using NLP + LSTM

This project is a machine learning-based Fake News Detection System that classifies news text (from social media or articles) as FAKE or REAL using a trained LSTM (Long Short-Term Memory) model, leveraging Natural Language Processing (NLP) and Attribution Scoring.

It allows users to upload datasets, preprocess text data, apply n-gram transformation, train the model, and test real-world news snippets.

---

📌 Features

- Dataset upload and preprocessing
- N-gram feature vectorization
- LSTM model training on news text
- Test prediction for new news samples
- Visualizations for Accuracy and Loss
- Output results shown and saved

---

📂 Folder Structure

Fake-News-Classifier/
├── Main.py                    # Main executable script
├── Output/
│   └── results.pdf            # Final prediction results
├── TwitterNewsData/
│   ├── news.csv               # Training dataset
│   └── testNews.txt           # Test data to classify
├── model/
│   ├── model.json             # Model architecture
│   ├── model_weights.h5       # Trained LSTM weights
│   ├── history.pckl           # Training history
│   └── model.txt              # Additional info
└── README.md

---

🧠 Algorithms & Techniques Used

- LSTM (Long Short-Term Memory) — for deep learning-based text classification
- TF-IDF & N-gram Analysis — feature extraction
- Named Entity Recognition (NER) — quote, verb, and entity extraction for attribution scoring
- Supervised Learning Estimator — for computing classification score

---

🧰 Technologies Used

Module         | Purpose
-------------- | -----------------------------------------------
TensorFlow     | LSTM Model & Deep Learning operations
NumPy          | Numerical operations
Pandas         | Data handling and CSV operations
Matplotlib     | Graph plotting (accuracy/loss)
Scikit-learn   | Preprocessing, model evaluation
re, os         | Regex + file ops
pickle         | Save/load model training history

---

📦 Installation

Make sure Python 3.7+ is installed. Then install dependencies:

pip install numpy pandas matplotlib scikit-learn tensorflow keras nltk

---

▶️ How to Run

1. Clone this repo or download the ZIP
2. Place news.csv in TwitterNewsData/
3. Run the main script:

python Main.py

4. Buttons will appear:
   - Upload Fake News Dataset → Load dataset
   - Preprocess Dataset & Apply NGram → Convert text to features
   - Run LSTM Algorithm → Train the model
   - Accuracy & Loss Graph → Show training performance
   - Test News Detection → Classify new news using testNews.txt

---

📊 Sample Output

- results.pdf in Output/ contains predictions:
  News Text --- PREDICTED AS: FAKE or GENUINE

---

✅ Results

- Achieved ~69.49% accuracy on LSTM
- Accuracy improves with higher-quality dataset and tuning
- Shows strong potential for real-world fake news filtering

---

🔮 Future Work

- Improve accuracy with larger, balanced datasets
- Add web interface (e.g., Flask or Streamlit)
- Include real-time Twitter scraping and analysis
- Multilingual support for global fake news detection

---

📚 Reference Paper

“A Taxonomy of Fake News Classification Techniques: Survey and Implementation Aspects”
This project is based on a full-spectrum research paper that explores fake news detection using NLP, supervised learning, and attribution-based scoring.
