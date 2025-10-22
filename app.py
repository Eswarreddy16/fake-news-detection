import streamlit as st
import joblib
import traceback
import os

# lazy imports for rebuild path
pd = None
TfidfVectorizer = None
LogisticRegression = None
train_test_split = None
classification_report = None

st.title("Fake News Detector")
st.write("Enter a News article below to check whether it is fake or real.")

# Load models with friendly errors
try:
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")
except Exception:
    st.error("Failed to load saved model or vectorizer. Make sure 'vectorizer.jb' and 'lr_model.jb' exist and are valid joblib files.")
    st.code(traceback.format_exc())
    st.stop()

# Defensive check: ensure the vectorizer is a fitted instance with a bound transform
def is_transform_bound_instance(obj):
    """Return True when obj.transform is a bound method on the instance (normal fitted vectorizer).

    This distinguishes between:
      - a fitted instance: obj.transform is a bound method with __self__ == obj
      - a class object: obj.transform is an unbound function (callable) but not bound to obj
    """
    try:
        m = getattr(obj, 'transform', None)
        if m is None:
            return False
        # bound method has a non-None __self__ attribute pointing to the instance
        if hasattr(m, '__self__') and m.__self__ is obj:
            return True
        # some objects might implement transform via __call__ or be callable objects; check common case
        return False
    except Exception:
        return False


if not is_transform_bound_instance(vectorizer):
    # Try to auto-rebuild a fitted vectorizer+model from available CSVs (best-effort)
    st.warning("Loaded vectorizer is not a fitted instance. Attempting to rebuild from data files (if available)...")
    rebuilt = False

    # attempt lazy imports only when needed
    try:
        import pandas as _pd
        from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
        from sklearn.linear_model import LogisticRegression as _LogisticRegression
        from sklearn.model_selection import train_test_split as _train_test_split
        from sklearn.metrics import classification_report as _classification_report

        pd = _pd
        TfidfVectorizer = _TfidfVectorizer
        LogisticRegression = _LogisticRegression
        train_test_split = _train_test_split
        classification_report = _classification_report
    except Exception:
        st.error("scikit-learn or pandas is not installed in the environment. Cannot attempt automatic rebuild.")
        st.info("Run the training script locally (python train_and_save.py) and then restart the app.")
        st.stop()

    # helper to find training data
    def find_training_data():
        # 1) combined CSV with text+label
        candidates = ['train.csv', 'data.csv', 'dataset.csv', 'real_and_fake.csv', 'combined.csv']
        for c in candidates:
            if os.path.exists(c):
                try:
                    df = pd.read_csv(c)
                    return df
                except Exception:
                    continue

        # 2) look for real.csv + Fake.csv
        if os.path.exists('real.csv') and os.path.exists('Fake.csv'):
            try:
                df_real = pd.read_csv('real.csv')
                df_fake = pd.read_csv('Fake.csv')
                # try to guess text column
                def pick_text_col(d):
                    for col in ['text', 'article', 'headline', d.columns[0]]:
                        if col in d.columns:
                            return col
                    return d.columns[0]
                tr = pick_text_col(df_real)
                tf = pick_text_col(df_fake)
                df_real2 = pd.DataFrame({'text': df_real[tr].astype(str)})
                df_real2['label'] = 1
                df_fake2 = pd.DataFrame({'text': df_fake[tf].astype(str)})
                df_fake2['label'] = 0
                df = pd.concat([df_real2, df_fake2], ignore_index=True)
                return df
            except Exception:
                return None

        # 3) try any csv in cwd and hope it has text+label
        for f in os.listdir('.'):
            if f.lower().endswith('.csv'):
                try:
                    df = pd.read_csv(f)
                    if 'text' in df.columns and 'label' in df.columns:
                        return df
                except Exception:
                    continue
        return None

    df = find_training_data()
    if df is None or df.shape[0] < 5:
        st.error("Could not find suitable training data files (e.g. 'real.csv' and 'Fake.csv', or a combined CSV with 'text' and 'label').")
        st.info("Please run 'python train_and_save.py' to create a fitted vectorizer and model, or provide a CSV with 'text' and 'label' columns.")
        st.stop()

    # prepare training data
    if 'text' not in df.columns or 'label' not in df.columns:
        # try to coerce
        df = df.rename(columns={df.columns[0]: 'text'})
        if 'label' not in df.columns:
            st.error("Found CSV but couldn't determine labels. Ensure CSV has 'text' and 'label' columns or provide 'real.csv' and 'Fake.csv'.")
            st.stop()

    X = df['text'].astype(str).tolist()
    y = df['label'].astype(int).tolist()

    # Fit vectorizer and a fresh model
    try:
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X_vec = vec.fit_transform(X)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_vec, y)

        # Save the newly fitted objects so next runs don't require rebuild
        joblib.dump(vec, 'vectorizer.jb')
        joblib.dump(clf, 'lr_model.jb')
        vectorizer = vec
        model = clf
        rebuilt = True
        st.success('Rebuilt and saved a fitted vectorizer and model from training files.')
    except Exception:
        st.error('Automatic rebuild failed due to an error during training. See traceback:')
        st.code(traceback.format_exc())
        st.stop()

    if not rebuilt:
        st.error("Vectorizer was not usable and automatic rebuild was not performed.")
        st.stop()

news_input = st.text_area("News Article:", "")

if st.button("Check News"):
    if news_input.strip():
        try:
            # Convert input to string and handle empty/None cases
            if not news_input or not news_input.strip():
                st.warning("Please enter some text to analyze.")
                st.stop()
                
            # Print debug info
            st.info("Processing input text...")
            
            # Transform the input text
            try:
                transform_input = vectorizer.transform([str(news_input).strip()])
            except Exception as e:
                st.error("Error during text transformation:")
                st.code(traceback.format_exc())
                st.stop()
                
            # Make prediction
            try:
                prediction = model.predict(transform_input)
                proba = model.predict_proba(transform_input)
            except Exception as e:
                st.error("Error during prediction:")
                st.code(traceback.format_exc())
                st.stop()
            
            # Show prediction with confidence
            try:
                fake_conf, real_conf = proba[0]
                
                # Show the analyzed text length and preview
                st.info(f"Analyzed text length: {len(news_input)} characters")
                st.info(f"Text preview: {news_input[:100]}...")
                
                if prediction[0] == 1:
                    st.success(f"📰 Prediction: The News is Real")
                    st.info(f"Confidence scores:\n"
                           f"- Real: {real_conf:.1%}\n"
                           f"- Fake: {fake_conf:.1%}")
                else:
                    st.error(f"🚫 Prediction: The News is Fake")
                    st.info(f"Confidence scores:\n"
                           f"- Fake: {fake_conf:.1%}\n"
                           f"- Real: {real_conf:.1%}")
                           
            except Exception as e:
                st.error("Error displaying results:")
                st.code(traceback.format_exc())
                
        except Exception as e:
            st.error("An unexpected error occurred:")
            st.code(traceback.format_exc())
    else:
        st.warning("Please enter some text to analyze.")