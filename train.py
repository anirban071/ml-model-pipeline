import pickle
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class ModelTrainer:
    def __init__(self, model_path: str = "model.pkl"):
        self.model_path = model_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def generate_data(self):
        # 4 features: e.g., age, income, transaction_amount, credit_score
        X, y = make_classification(n_samples=1000, n_features=4, random_state=42)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_and_save(self):
        X_train, X_test, y_train, y_test = self.generate_data()
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        score = self.model.score(X_test, y_test)
        print(f"Model accuracy: {score:.2f}")

        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {self.model_path}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_and_save()