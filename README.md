# Explainable Fake News Detection System

An end-to-end Machine Learning project using **XAI (Explainable AI)** to detect disinformation.

## 🚀 Execution Guide
1. **Clone & Setup:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Data Preparation:**
   - Download the [Kaggle Fake & Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
   - Place `True.csv` and `Fake.csv` in `data/raw/`.
3. **Train the Model:**
   Create a small script `main.py` calling `FakeNewsModel.train()`.
4. **Run Web App:**
   ```bash
   streamlit run app.py
   ```

## ☁️ Deployment (Streamlit Cloud)
1. Push this folder to a GitHub repository.
2. Sign in to [Streamlit Cloud](https://share.streamlit.io/).
3. Click "New App", select your repo, and point to `app.py`.
4. Add `nltk.download` commands to a `packages.txt` or handle them in `app.py` (already handled in our code).

## 🔬 Research-Level Improvements

### 1. Transformer-based Models (BERT)
While Logistic Regression is interpretable, **BERT (Bidirectional Encoder Representations from Transformers)** captures context. 
- **Advantage:** Detects nuance and sequence.
- **Explainability:** Use **Integrated Gradients** or **Attention Maps** instead of SHAP to see which parts of the sentence the transformer is "attending" to.

### 2. SHAP vs LIME Consistency
Research suggests that SHAP and LIME do not always agree. 
- **Research Question:** "To what extent do local explanations from LIME and SHAP provide consistent feature importance rankings in short-form disinformation?"
- **Test:** Run both on the same 100 articles and calculate the **Kendall's Tau correlation** between their feature rankings.

### 3. Turning this into a Research Paper
To make this research-grade:
- **Cross-Domain Testing:** Train on political news, test on health/COVID news. Observe how explainability highlights domain shift.
- **Human-in-the-loop Evaluation:** Conduct a study where users are shown (A) just the prediction and (B) the prediction with SHAP. Measure if SHAP actually increases user's ability to spot a "False Positive."
- **Adversarial Robustness:** Test how adding "trigger words" changes the explanation without changing the prediction.
