import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from src.preprocessing import TextPreprocessor
import os

class FakeNewsModel:
    def __init__(self, model_path='models/model.pkl', vec_path='models/vectorizer.pkl'):
        self.model_path = model_path
        self.vec_path = vec_path
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=1000)
        self.preprocessor = TextPreprocessor()

    def prepare_data(self, true_csv, fake_csv):
        # Load and label
        df_true = pd.read_csv(true_csv)
        df_fake = pd.read_csv(fake_csv)
        df_true['target'] = 1  # Real
        df_fake['target'] = 0  # Fake
        
        df = pd.concat([df_true, df_fake]).reset_index(drop=True)
        # Combine title and text
        df['total_text'] = df['title'] + " " + df['text']
        
        print("Cleaning text (this may take a while)...")
        df['total_text'] = df['total_text'].apply(self.preprocessor.clean_text)
        return df

    def train(self, df):
        X_train, X_test, y_train, y_test = train_test_split(
            df['total_text'], df['target'], test_size=0.2, random_state=42
        )
        
        print("Vectorizing...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print("Training Logistic Regression...")
        self.model.fit(X_train_tfidf, y_train)
        
        # Build evaluation metrics
        y_pred = self.model.predict(X_test_tfidf)
        y_prob = self.model.predict_proba(X_test_tfidf)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "report": classification_report(y_test, y_pred),
            "cm": confusion_matrix(y_test, y_pred)
        }
        
        # Save artifacts
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vec_path)
        
        return metrics

    def predict(self, raw_text):
        cleaned = self.preprocessor.clean_text(raw_text)
        vec = self.vectorizer.transform([cleaned])
        prob = self.model.predict_proba(vec)[0]
        return prob # [prob_fake, prob_real]
