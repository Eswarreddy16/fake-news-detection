# Fake News Detector

This small app demonstrates a TF-IDF + Logistic Regression fake news detector.

Files:
- `app.py` - Streamlit UI (loads `vectorizer.jb` and `lr_model.jb`)
- `train_and_save.py` - Example training script that fits a TfidfVectorizer and LogisticRegression, then saves them
- `inspect_vectorizer.py` - Inspect what's stored in `vectorizer.jb`

Setup & run (Windows PowerShell)

1. Create and activate venv (if you don't have one)
   python -m venv venv
   .\\venv\\Scripts\\Activate.ps1

2. Install requirements
   pip install -r requirements.txt

3. Train and save (or place your own `vectorizer.jb` and `lr_model.jb`)
   python train_and_save.py

4. (Optional) Inspect saved vectorizer
   python inspect_vectorizer.py

5. Run the Streamlit app
   streamlit run app.py
