import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NewsExplainer:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        # Robust explainer for scikit-learn pipeline
        self.predict_fn = lambda x: self.model.predict_proba(self.vectorizer.transform(x))[:, 1]
        self.explainer = shap.Explainer(self.predict_fn, masker=shap.maskers.Text(tokenizer=r"\W+"))

    def get_local_explanation(self, text_list):
        """
        Explains a specific instance of news.
        """
        try:
            shap_values = self.explainer(text_list)
            return shap_values
        except Exception as e:
            print(f"SHAP Error: {e}")
            return None

    def get_linguistic_audit(self, text):
        """
        Advanced Linguistic AI Agent
        Analyzes syntactic and stylistic patterns.
        """
        if not text:
            return {"status": "error", "report": ["No input provided."]}
            
        words = text.split()
        word_count = len(words)
        caps_words = [w for w in words if w.isupper() and len(w) > 1]
        exclamations = text.count('!')
        question_marks = text.count('?')
        
        # Scoring logic
        sensationalism_score = (len(caps_words) / (word_count + 1)) * 100
        urgency_score = (exclamations / (word_count + 1)) * 100
        inquiry_ratio = (question_marks / (word_count + 1)) * 100
        
        report = []
        if sensationalism_score > 15:
            report.append(f"⚠️ HIGH SENSATIONALISM: {len(caps_words)} words are in ALL CAPS. This is a common tactic in clickbait.")
        if urgency_score > 5:
            report.append(f"🚨 URGENCY MARKERS: High frequency of exclamation marks ({exclamations}). Often used to provoke emotional reactions.")
        if inquiry_ratio > 5:
            report.append("❓ SPECULATIVE TONE: High number of question marks. May indicate 'Loaded Questions' or unverified claims.")
        if word_count < 30:
            report.append("📉 LOW SUBSTANCE: The article is very short. Legitimate news usually provides more context.")
            
        final_assessment = "Neutral"
        if len(report) >= 2:
            final_assessment = "Suspicious"
        elif len(report) == 1:
            final_assessment = "Caution Advised"
            
        return {
            "assessment": final_assessment,
            "report": report if report else ["Linguistic patterns appear standard and professional."],
            "stats": {
                "Sensationalism": f"{sensationalism_score:.1f}%",
                "Urgency": f"{urgency_score:.1f}%",
                "Speculation": f"{inquiry_ratio:.1f}%"
            }
        }

    def get_related_intel(self, query_text, dataset_df):
        """
        Uses cosine similarity to find highly related news from the audit archives.
        Increased sample size and improved text fusion for better matching.
        """
        if not query_text or dataset_df is None:
            return []
            
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Preprocess query
        query_vec = self.vectorizer.transform([str(query_text)])
        
        # Improve coverage by sampling more records (balanced for performance)
        sample_size = min(len(dataset_df), 3000)
        subset = dataset_df.sample(sample_size, random_state=42)
        
        # Vectorize dataset
        dataset_vecs = self.vectorizer.transform(subset['total_text'].astype(str))
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, dataset_vecs).flatten()
        
        # Get top 5 matches (increased from 3 for better coverage)
        top_indices = similarities.argsort()[-5:][::-1]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score < 0.05: continue # Filter out completely irrelevant results
            
            row = subset.iloc[idx]
            results.append({
                "title": str(row['title']).strip(),
                "content": str(row['text']).strip()[:350] + "...",
                "similarity": f"{score:.1%}",
                "status": "Verified" if row['target'] == 1 else "Flagged",
                "label": "REAL" if row['target'] == 1 else "FAKE"
            })
        return results

    def get_global_importance(self):
        """
        Returns top features for Real vs Fake based on model coefficients.
        """
        coefs = self.model.coef_[0]
        features = self.vectorizer.get_feature_names_out()
        
        importance_df = pd.DataFrame({'word': features, 'coefficient': coefs})
        
        real_indicators = importance_df.sort_values(by='coefficient', ascending=False).head(20).copy()
        fake_indicators = importance_df.sort_values(by='coefficient', ascending=True).head(20).copy()
        
        # Make them look nicer for the UI
        real_indicators.columns = ['Vibrant Word', 'Credibility Weight']
        fake_indicators.columns = ['Suspicious Word', 'Deception Weight']
        
        return real_indicators, fake_indicators
