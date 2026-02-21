from src.training import FakeNewsModel
from src.data_generator import create_sample_data
import os

def main():
    if not os.path.exists('data/raw/True.csv') or not os.path.exists('data/raw/Fake.csv'):
        print("Real data not found. Generating sample data for demonstration...")
        create_sample_data()
    
    print("Initializing Training Pipeline...")
    model = FakeNewsModel()
    
    print("Loading and Preprocessing Data...")
    df = model.prepare_data('data/raw/True.csv', 'data/raw/Fake.csv')
    
    print("Training Model...")
    metrics = model.train(df)
    
    print("\nTraining Complete!")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(metrics['report'])

if __name__ == "__main__":
    main()
