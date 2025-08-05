
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
class IrisClassificationPipeline:
    
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None  
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_model_name = None
        
    def load_and_explore_data(self):
        """Load the iris dataset and take a look at what we're working with"""
        print("\n=== Step 1: Loading and exploring the data ===")
        
        try:
            self.df = pd.read_csv('iris.csv')
            print("Found iris.csv file - loading from there")
        except FileNotFoundError:
            print("No CSV found, using sklearn's built-in dataset")
            iris = load_iris()
            self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
            self.df['species'] = iris.target_names[iris.target]
        
        print(f"\nOkay, loaded the data! Here's what I'm working with:")
        print(f"   Dataset shape: {self.df.shape}")
        print(f"   Features: {list(self.df.columns[:-1])}")
        print(f"   Target variable: {self.df.columns[-1]}")
        
        print(f"\nData quality check:")
        missing_vals = self.df.isnull().sum().sum()
        duplicates = self.df.duplicated().sum()
        print(f"   Missing values: {missing_vals} {'(good!)' if missing_vals == 0 else '(need to fix)'}")
        print(f"   Duplicate rows: {duplicates} {'(clean data!)' if duplicates == 0 else '(might need cleaning)'}")
        
        print(f"\nSpecies breakdown:")
        for species, count in self.df['species'].value_counts().items():
            print(f"   {species}: {count} flowers")
        
        print(f"\nSome basic statistics:")
        print(self.df.describe().round(2))
        
        return self.df
    
    def create_visualizations(self):
        """Create some nice plots to understand the data better"""
        print("\n=== Step 2: Making some visualizations ===")
        print("Time to see what this data actually looks like...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Iris Dataset - Visual Analysis', fontsize=16, fontweight='bold')
        
        feature_cols = [col for col in self.df.columns if col != 'species']
        
        for species in self.df['species'].unique():
            species_data = self.df[self.df['species'] == species]
            axes[0, 0].scatter(species_data.iloc[:, 0], species_data.iloc[:, 1], 
                              label=species, alpha=0.7, s=60)
        axes[0, 0].set_xlabel(feature_cols[0])
        axes[0, 0].set_ylabel(feature_cols[1])
        axes[0, 0].set_title('Sepal Measurements')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        for species in self.df['species'].unique():
            species_data = self.df[self.df['species'] == species]
            axes[0, 1].scatter(species_data.iloc[:, 2], species_data.iloc[:, 3], 
                              label=species, alpha=0.7, s=60)
        axes[0, 1].set_xlabel(feature_cols[2])
        axes[0, 1].set_ylabel(feature_cols[3])
        axes[0, 1].set_title('Petal Measurements')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        species_counts = self.df['species'].value_counts()
        axes[0, 2].pie(species_counts.values, labels=species_counts.index, 
                       autopct='%1.1f%%', startangle=90)
        axes[0, 2].set_title('How many of each species?')
        
        correlation_matrix = self.df[feature_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                   center=0, ax=axes[1, 0])
        axes[1, 0].set_title('Feature Correlations')
        
        self.df.boxplot(column=feature_cols[0], by='species', ax=axes[1, 1])
        axes[1, 1].set_title(f'{feature_cols[0]} by Species')
        axes[1, 1].set_xlabel('Species')
        
        for species in self.df['species'].unique():
            species_data = self.df[self.df['species'] == species][feature_cols[0]]
            axes[1, 2].hist(species_data, alpha=0.7, label=species, bins=15)
        axes[1, 2].set_xlabel(feature_cols[0])
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title(f'{feature_cols[0]} Distribution')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('iris_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Created some nice plots! Saved as 'iris_analysis_comprehensive.png'")
        print("The petal measurements look like they'll be really useful for classification")
    
    def preprocess_data(self):
        """Get the data ready for machine learning"""
        print("\n=== Step 3: Preparing the data ===")
        print("Now I need to clean this up and get it ready for the algorithms...")
        
        feature_columns = [col for col in self.df.columns if col != 'species']
        X = self.df[feature_columns]
        y = self.df['species']
        
        print(f"Separated the data:")
        print(f"   Features: {feature_columns}")
        print(f"   Target: '{y.name}' (what we want to predict)")
        print(f"   X shape: {X.shape}, y shape: {y.shape}")
        
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"\nConverted species names to numbers:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"   '{class_name}' -> {i}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\nSplit the data (80% train, 20% test):")
        print(f"   Training set: {self.X_train.shape[0]} samples")
        print(f"   Test set: {self.X_test.shape[0]} samples")
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"   Applied feature scaling (StandardScaler)")
        print("Data preprocessing done! Ready to train some models.")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """Train a bunch of different models and see which one works best"""
        print("\n=== Step 4: Training the models ===")
        print("Time for the fun part - let's see which algorithm wins!")
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
        }
        
        print("Training these models:")
        for name, model in models.items():
            print(f"   Training {name}...", end=" ")
            
            model.fit(self.X_train_scaled, self.y_train)
            self.models[name] = model
            
            y_pred = model.predict(self.X_test_scaled)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            
            self.results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"Test accuracy: {accuracy:.4f}")
        
        self.best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🏆 Winner: {self.best_model_name}!")
        print(f"   Test accuracy: {self.results[self.best_model_name]['accuracy']:.4f}")
        print(f"   CV score: {self.results[self.best_model_name]['cv_mean']:.4f} ± {self.results[self.best_model_name]['cv_std']:.4f}")
        
        return self.models, self.results
    
    def evaluate_models(self):
        """Detailed model evaluation with metrics and visualizations"""
        print("\nSTEP 5: DETAILED MODEL EVALUATION")
        print("-" * 40)
        
        # Create evaluation plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
        
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        
        bars = axes[0, 0].bar(range(len(model_names)), accuracies, 
                             color=['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].set_xticks(range(len(model_names)))
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].set_ylim(0.8, 1.0)
        
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', 
                           ha='center', va='bottom', fontweight='bold')
        
        cm = confusion_matrix(self.y_test, self.results[self.best_model_name]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   ax=axes[0, 1])
        axes[0, 1].set_title(f'Confusion Matrix - {self.best_model_name}')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        cv_means = [self.results[name]['cv_mean'] for name in model_names]
        cv_stds = [self.results[name]['cv_std'] for name in model_names]
        
        axes[1, 0].bar(range(len(model_names)), cv_means, yerr=cv_stds,
                      capsize=5, alpha=0.7, color='lightblue')
        axes[1, 0].set_title('Cross-Validation Scores')
        axes[1, 0].set_ylabel('CV Accuracy')
        axes[1, 0].set_xticks(range(len(model_names)))
        axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
        
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            feature_importance = rf_model.feature_importances_
            feature_names = [col for col in self.df.columns if col != 'species']
            
            bars = axes[1, 1].bar(range(len(feature_names)), feature_importance, 
                                 color='lightgreen')
            axes[1, 1].set_title('Feature Importance (Random Forest)')
            axes[1, 1].set_ylabel('Importance Score')
            axes[1, 1].set_xticks(range(len(feature_names)))
            axes[1, 1].set_xticklabels(feature_names, rotation=45, ha='right')
            
            for i, (bar, imp) in enumerate(zip(bars, feature_importance)):
                axes[1, 1].text(i, imp + 0.01, f'{imp:.3f}', 
                               ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nDetailed Classification Report ({self.best_model_name}):")
        print(classification_report(self.y_test, 
                                  self.results[self.best_model_name]['predictions'],
                                  target_names=self.label_encoder.classes_))
        
        print("Model evaluation completed and saved as 'model_evaluation_results.png'")
    
    def save_model(self):
        """Save the best model and preprocessing components"""
        print("\nSTEP 6: MODEL PERSISTENCE")
        print("-" * 40)
        
        model_filename = f'best_iris_model_{self.best_model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.joblib'
        
        joblib.dump(self.best_model, model_filename)
        joblib.dump(self.scaler, 'iris_feature_scaler.joblib')
        joblib.dump(self.label_encoder, 'iris_label_encoder.joblib')
        
        print(f"Model artifacts saved:")
        print(f"   • Best model: {model_filename}")
        print(f"   • Feature scaler: iris_feature_scaler.joblib")
        print(f"   • Label encoder: iris_label_encoder.joblib")
        
        return model_filename
    
    def predict_new_samples(self):
        """Demonstrate predictions on new data"""
        print("\nSTEP 7: PREDICTION DEMONSTRATION")
        print("-" * 40)
        
        test_samples = [
            {
                "name": "Small Flower (Typical Setosa)",
                "measurements": [4.5, 3.0, 1.3, 0.2],
                "expected": "setosa"
            },
            {
                "name": "Medium Flower (Typical Versicolor)", 
                "measurements": [5.8, 2.7, 4.1, 1.3],
                "expected": "versicolor"
            },
            {
                "name": "Large Flower (Typical Virginica)",
                "measurements": [6.8, 3.2, 5.9, 2.3], 
                "expected": "virginica"
            },
            {
                "name": "Edge Case Sample",
                "measurements": [5.5, 3.5, 2.0, 0.8],
                "expected": "unknown"
            }
        ]
        
        print("Testing Model Predictions:")
        
        for sample in test_samples:
            measurements = sample["measurements"]
            sepal_l, sepal_w, petal_l, petal_w = measurements
            
            sample_array = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
            sample_scaled = self.scaler.transform(sample_array)
            
            prediction = self.best_model.predict(sample_scaled)[0]
            probabilities = self.best_model.predict_proba(sample_scaled)[0]
            
            species = self.label_encoder.inverse_transform([prediction])[0]
            confidence = max(probabilities)
            
            print(f"\n{sample['name']}:")
            print(f"   Measurements: Sepal(L={sepal_l}, W={sepal_w}) | Petal(L={petal_l}, W={petal_w})")
            print(f"   Predicted: {species.upper()} (Confidence: {confidence:.4f})")
            if sample['expected'] != "unknown":
                status = "CORRECT" if species == sample['expected'] else "INCORRECT"
                print(f"   Expected: {sample['expected'].upper()} - {status}")
        
        print("\nPrediction demonstration completed")
    
    def generate_summary_report(self):
        print("\nPROJECT SUMMARY REPORT")
        print("=" * 60)
        
        print("\nMODEL PERFORMANCE COMPARISON:")
        print("-" * 50)
        print(f"{'Model':<25} {'Accuracy':<10} {'CV Score':<15}")
        print("-" * 50)
        
        for name in self.results:
            acc = self.results[name]['accuracy']
            cv = self.results[name]['cv_mean']
            cv_std = self.results[name]['cv_std']
            marker = "BEST " if name == self.best_model_name else "     "
            print(f"{marker}{name:<20} {acc:<10.4f} {cv:.4f}±{cv_std:.4f}")
        
        print("\nPROJECT ACHIEVEMENTS:")
        print(f"   Dataset: 150 samples, 4 features, 3 classes")
        print(f"   Best Model: {self.best_model_name}")
        print(f"   Best Accuracy: {self.results[self.best_model_name]['accuracy']:.4f}")
        print(f"   Models Trained: {len(self.models)}")
        print(f"   Visualizations: 2 comprehensive plots created")
        print(f"   Model Deployment: Ready for production use")
        
        print("\nKEY SKILLS DEMONSTRATED:")
        skills = [
            "Data Loading and Exploration",
            "Statistical Analysis and Visualization", 
            "Data Preprocessing and Feature Scaling",
            "Multiple ML Algorithm Implementation",
            "Model Evaluation and Cross-Validation",
            "Performance Comparison and Selection",
            "Model Persistence and Deployment",
            "Prediction System Development"
        ]
        
        for skill in skills:
            print(f"   {skill}")
        
        print("\n" + "=" * 60)
        print("IRIS CLASSIFICATION PROJECT COMPLETED SUCCESSFULLY!")
        print("Ready for internship portfolio submission")



def main():
    
    
    pipeline = IrisClassificationPipeline()
    
    try:
        print("Starting the iris classification pipeline...")
        
        pipeline.load_and_explore_data()
        pipeline.create_visualizations()
        pipeline.preprocess_data()
        pipeline.train_models()
        pipeline.evaluate_models()
        pipeline.save_model()
        pipeline.predict_new_samples()
        pipeline.generate_summary_report()
        
        print(f"\nEverything worked! Pipeline completed successfully.")
        print(f"Check out these files I created:")
        print(f"iris_analysis_comprehensive.png (data visualizations)")
        print(f"model_evaluation_results.png (model comparison charts)") 
        print(f"joblib files (saved models)")
        
    except Exception as e:
        print(f"\nOops, something went wrong: {e}")
        print(f"Let me know if you see this error - I might need to debug!")
    
    return pipeline
def load_trained_model():
    """Load saved model for deployment"""
    try:
        model = joblib.load('best_iris_model_svm_rbf.joblib')
        scaler = joblib.load('iris_feature_scaler.joblib')
        encoder = joblib.load('iris_label_encoder.joblib')
        return model, scaler, encoder
    except FileNotFoundError:
        print("Model files not found. Please run the main pipeline first.")
        return None, None, None

def predict_iris_flower(sepal_length, sepal_width, petal_length, petal_width):
    model, scaler, encoder = load_trained_model()
    
    if model is None:
        return {"error": "Couldn't load the model - did you train it first?"}
    
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    sample_scaled = scaler.transform(sample)
    
    prediction = model.predict(sample_scaled)[0]
    probabilities = model.predict_proba(sample_scaled)[0]
    
    species = encoder.inverse_transform([prediction])[0]
    confidence = max(probabilities)
    
    return {
        "species": species,
        "confidence": float(confidence),
        "all_probabilities": {
            cls: float(prob) for cls, prob in zip(encoder.classes_, probabilities)
        }
    }



def get_user_input_prediction():
    """Get flower measurements from user input and make a prediction"""
    print("\n=== Interactive Prediction ===")
    print("Enter the measurements of the iris flower you want to classify:")
    
    try:
        sepal_length = float(input("Sepal Length (cm): "))
        sepal_width = float(input("Sepal Width (cm): "))
        petal_length = float(input("Petal Length (cm): "))
        petal_width = float(input("Petal Width (cm): "))
        
        print(f"\nMaking prediction for:")
        print(f"  Sepal: {sepal_length}cm x {sepal_width}cm")
        print(f"  Petal: {petal_length}cm x {petal_width}cm")
        
        result = predict_iris_flower(sepal_length, sepal_width, petal_length, petal_width)
        
        if "error" not in result:
            print(f"\nPrediction: {result['species'].upper()}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"\nAll probabilities:")
            for species, prob in result['all_probabilities'].items():
                print(f"  {species}: {prob:.1%}")
        else:
            print(f"Error: {result['error']}")
            
    except ValueError:
        print("Invalid input! Please enter numeric values only.")
    except KeyboardInterrupt:
        print("\nPrediction cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    print("Starting my iris classification project...")
    iris_pipeline = main()
    
    print(f"\n=== Testing the prediction function ===")
    print("Let me try predicting with some sample measurements...")
    
    result = predict_iris_flower(5.1, 3.5, 1.4, 0.2)
    if "error" not in result:
        print(f"Prediction: {result['species'].upper()}")
        print(f"Confidence: {result['confidence']:.1%}")
        print("Looks like the model is working! 🎉")
    else:
        print(f"Hmm, got an error: {result['error']}")
    while True:
        print("\n" + "="*50)
        choice = input("\nWould you like to make your own prediction? (y/n): ").lower().strip()
        
        if choice in ['y', 'yes']:
            get_user_input_prediction()
        elif choice in ['n', 'no']:
            print("Thanks for using the iris classifier!")
            break
        else:
            print("Please enter 'y' for yes or 'n' for no.")
