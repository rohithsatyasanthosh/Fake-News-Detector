import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Pre-compile regex for efficiency
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
PUNCT_PATTERN = re.compile('[%s]' % re.escape(string.punctuation))
NUM_PATTERN = re.compile(r'\d+')

class TextPreprocessor:
    def __init__(self):
        # Cache resources locally
        resources = ['stopwords', 'wordnet', 'omw-1.4', 'punkt']
        for res in resources:
            try:
                nltk.data.find(f'corpora/{res}' if res != 'punkt' else f'tokenizers/{res}')
            except LookupError:
                nltk.download(res, quiet=True)
            
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        if not text or not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        # Regex removals
        text = URL_PATTERN.sub('', text)
        text = PUNCT_PATTERN.sub('', text)
        text = NUM_PATTERN.sub('', text)
        
        # Faster tokenization for simple cleaning
        tokens = text.split() 
        
        # Lemmatize and remove stop words in one list comprehension
        cleaned_tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens if word not in self.stop_words
        ]
        return " ".join(cleaned_tokens)
