# train_and_save.py
# Train a simple TF-IDF + LogisticRegression classifier and save the fitted vectorizer and model.
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# Loads and combines real.csv (label=1) and Fake.csv (label=0)
# Each CSV should have a text column (will try to detect it: 'text', 'article', 'content', 'title' or first column)

def load_example_data():
    # Try to load real.csv + Fake.csv
    try:
        print("Loading real.csv and Fake.csv...")
        
        # Load real news (label 1)
        df_real = pd.read_csv('real.csv')
        # Try to find the text column (common names or first column)
        real_text_col = None
        for col in ['text', 'article', 'content', 'title']:
            if col in df_real.columns:
                real_text_col = col
                break
        if real_text_col is None:
            real_text_col = df_real.columns[0]  # use first column
        print(f"Using column '{real_text_col}' from real.csv")
        
        # Load fake news (label 0)
        df_fake = pd.read_csv('Fake.csv')
        # Try to find the text column
        fake_text_col = None
        for col in ['text', 'article', 'content', 'title']:
            if col in df_fake.columns:
                fake_text_col = col
                break
        if fake_text_col is None:
            fake_text_col = df_fake.columns[0]  # use first column
        print(f"Using column '{fake_text_col}' from Fake.csv")
        
        # Create combined dataset with text and label columns
        df_real_prep = pd.DataFrame({
            'text': df_real[real_text_col].astype(str),
            'label': 1
        })
        df_fake_prep = pd.DataFrame({
            'text': df_fake[fake_text_col].astype(str),
            'label': 0
        })
        
        # Combine real and fake news
        df = pd.concat([df_real_prep, df_fake_prep], ignore_index=True)
        
        # Shuffle thoroughly
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        print(f"Combined dataset: {len(df)} rows ({len(df_real_prep)} real, {len(df_fake_prep)} fake)")
        
        # Print some random examples to verify labels
        print("\nSample real news (label=1):")
        real_sample = df[df['label'] == 1].sample(n=2)
        for _, row in real_sample.iterrows():
            print(f"\n{row['text'][:200]}...")
            
        print("\nSample fake news (label=0):")
        fake_sample = df[df['label'] == 0].sample(n=2)
        for _, row in fake_sample.iterrows():
            print(f"\n{row['text'][:200]}...")
        
        return df
        
    except Exception as e:
        print(f"Could not load real.csv and Fake.csv ({str(e)}). Using small example dataset instead.")
        texts = [
            "The stock market saw a huge gain today as investors cheered earnings.",
            "Aliens landed in my backyard and gave me advice about investing.",
            "Local team wins championship after close final match.",
            "Miracle cure for all diseases discovered in remote village!"
        ]
        labels = [1, 0, 1, 0]
        df = pd.DataFrame({"text": texts, "label": labels})
        return df


def main():
    df = load_example_data()
    if 'text' not in df.columns or 'label' not in df.columns:
        raise RuntimeError("Dataframe must have 'text' and 'label' columns. Edit train_and_save.py to load your data.")

    X = df['text'].astype(str).tolist()
    y = df['label'].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(set(y))>1 else None)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    X_test_vec = vectorizer.transform(X_test)
    preds = model.predict(X_test_vec)
    print("Evaluation on test split:")
    print(classification_report(y_test, preds))

    print("\nSaving fitted vectorizer and model...")
    joblib.dump(vectorizer, 'vectorizer.jb')
    joblib.dump(model, 'lr_model.jb')
    print("✓ Saved vectorizer.jb")
    print("✓ Saved lr_model.jb")

if __name__ == '__main__':
    main()
