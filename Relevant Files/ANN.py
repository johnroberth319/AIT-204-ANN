import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from itertools import combinations
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))

class NeuralNetwork:
    """
    Neural network for predicting basketball team quality scores.
    """
    
    def __init__(self, input_dim=21):
        self.input_dim = input_dim
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def build_model(self, hidden_layers=[64, 32, 16], dropout_rate=0.3, learning_rate=0.001):
        """
        Build the neural network architecture.
        
        Args:
            hidden_layers: List of neurons in each hidden layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(self.input_dim,)))
        
        # Hidden layers with dropout for regularization
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(units, activation='relu', name=f'hidden_{i+1}'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer (single neuron for team score prediction)
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        print("Neural Network Architecture:")
        print(self.model.summary())
        return model
    
    def prepare_data(self, features_file="team_features.csv", test_size=0.2, random_state=42):
        """
        Load and prepare data for training.
        
        Args:
            features_file: Path to team features CSV
            test_size: Fraction of data for testing
            random_state: Random seed for reproducible splits
        """
        
        # Load features
        df = pd.read_csv(os.path.join(script_dir, features_file))
        print(f"Loaded {len(df)} team examples")
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col != 'team_score']
        X = df[feature_columns].values
        y = df['team_score'].values
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Score range: {y.min():.3f} - {y.max():.3f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.feature_columns = feature_columns
        
        print(f"Training samples: {len(X_train_scaled)}")
        print(f"Testing samples: {len(X_test_scaled)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train, X_test, y_test, 
                   epochs=30, batch_size=32, validation_split=0.2, patience=10):
        """
        Train the neural network.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data  
            epochs: Maximum training epochs
            batch_size: Training batch size
            validation_split: Fraction for validation during training
            patience: Early stopping patience
        """
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        print(f"Training neural network...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        test_loss, test_mae, test_mse = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Make predictions
        y_pred = self.model.predict(X_test, verbose=0).flatten()
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nFinal Test Results:")
        print(f"  Test Loss (MSE): {test_loss:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Test R²: {r2:.4f}")
        
        return self.history
    
    def plot_training_history(self, save_plot=True):
        """Plot training history."""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss', alpha=0.8)
        ax1.plot(self.history.history['val_loss'], label='Validation Loss', alpha=0.8)
        ax1.set_title('Model Loss During Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE plot
        ax2.plot(self.history.history['mae'], label='Training MAE', alpha=0.8)
        ax2.plot(self.history.history['val_mae'], label='Validation MAE', alpha=0.8)
        ax2.set_title('Model MAE During Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(script_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
            print("Training history plot saved as 'training_history.png'")
        
        plt.show()
    
    def save_model(self, model_path="basketball_team_model.keras", scaler_path="feature_scaler.pkl"):
        """Save trained model and scaler."""
        if self.model is None:
            print("No model to save.")
            return
        
        # Save model
        model_full_path = os.path.join(script_dir, model_path)
        self.model.save(model_full_path)
        
        # Save scaler
        scaler_full_path = os.path.join(script_dir, scaler_path)
        with open(scaler_full_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model saved to: {model_full_path}")
        print(f"Scaler saved to: {scaler_full_path}")
    
    def load_model(self, model_path="basketball_team_model.keras", scaler_path="feature_scaler.pkl"):
        """Load trained model and scaler."""
        
        # Load model
        model_full_path = os.path.join(script_dir, model_path)
        self.model = keras.models.load_model(model_full_path)
        
        # Load scaler
        scaler_full_path = os.path.join(script_dir, scaler_path)
        with open(scaler_full_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"Model loaded from: {model_full_path}")
        print(f"Scaler loaded from: {scaler_full_path}")
    
    def predict_team_score(self, team_features):
        """
        Predict team score from features.
        
        Args:
            team_features: List or array of team features
        
        Returns:
            Predicted team score (0-1)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train a model first.")
        
        # Ensure correct shape
        features = np.array(team_features).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled, verbose=0)[0, 0]
        
        return prediction