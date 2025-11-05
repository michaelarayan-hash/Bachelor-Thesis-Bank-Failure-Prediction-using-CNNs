"""
Final Training and Testing Phase for Bank Failure Prediction

This script trains models to optimal epochs (from initial training) and
performs comprehensive testing with metrics and confusion matrices.
"""

import os
import pickle
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from typing import Dict, Tuple
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


class CNNModel:
    """Handles CNN model creation."""
    
    @staticmethod
    def build_cnn(input_shape: Tuple, kernel_dim: int = 3) -> keras.Model:
        """
        Build CNN model for binary classification.
        
        Args:
            input_shape: Shape of input data (height, width, channels)
            kernel_dim: Kernel dimension for convolutional layers
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (kernel_dim, kernel_dim), activation='relu', padding='same'),
            layers.Conv2D(64, (kernel_dim, kernel_dim), activation='relu', padding='same'),
            layers.Conv2D(128, (kernel_dim, kernel_dim), activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=7e-7)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        return model


class DataProcessor:
    """Handles data loading and preparation."""
    
    @staticmethod
    def prepare_data(
        dead_dict: Dict,
        alive_dict: Dict,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert dictionaries to image arrays and split into train/test.
        
        Args:
            dead_dict: Dictionary of dead bank arrays
            alive_dict: Dictionary of alive bank arrays
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X, y = [], []
        
        # Add dead bank data (label = 0)
        for key, img in dead_dict.items():
            X.append(img)
            y.append(0)
        
        # Add alive bank data (label = 1)
        for key, img in alive_dict.items():
            X.append(img)
            y.append(1)
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for CNN: (samples, height, width, channels)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        X = X.astype('float64')
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def load_pickle(filepath: str) -> Dict:
        """Load pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def extract_key(filepath: str) -> str:
        """Extract clean identifier from filename."""
        filename = os.path.basename(filepath)
        key = filename.replace("dead_", "").replace("alive_", "").replace("bank_", "").replace(".pickle", "")
        return key


class FinalTrainer:
    """Handles final model training and comprehensive evaluation."""
    
    def __init__(self, output_dir: str):
        """
        Initialize trainer.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
    
    def train_and_evaluate(
        self,
        name: str,
        dead_dict: Dict,
        alive_dict: Dict,
        epochs: int,
        batch_size: int = 64
    ) -> Dict:
        """
        Train model to specified epochs and evaluate comprehensively.
        
        Args:
            name: Name of the variable set
            dead_dict: Dictionary of dead bank arrays
            alive_dict: Dictionary of alive bank arrays
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Training and evaluating: {name}")
        print(f"Target epochs: {epochs}")
        print(f"{'='*60}")
        
        # Prepare data
        X_train, X_test, y_train, y_test = DataProcessor.prepare_data(
            dead_dict, alive_dict
        )
        
        print(f"Data shapes:")
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
        
        # Build model
        input_shape = X_train.shape[1:]
        model = CNNModel.build_cnn(input_shape, kernel_dim=3)
        
        # Train model
        print(f"\nTraining for {epochs} epochs...")
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions
        y_pred_probs = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_pred_probs >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_probs)
        metrics['Name'] = name
        metrics['Epochs'] = epochs
        metrics['Test Loss'] = test_loss
        
        print(f"\n{name} Results:")
        print(f"  Test Accuracy: {metrics['Test Accuracy']:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  ROC AUC: {metrics['ROC AUC']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall: {metrics['Recall']:.4f}")
        print(f"  F1 Score: {metrics['F1 Score']:.4f}")
        
        # Save results
        self.results.append(metrics)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_test, y_pred, name)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['Dead Bank (0)', 'Alive Bank (1)'],
            digits=4
        ))
        
        return metrics
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_probs: np.ndarray
    ) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        return {
            'Test Accuracy': accuracy_score(y_true, y_pred),
            'ROC AUC': roc_auc_score(y_true, y_pred_probs),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1 Score': f1_score(y_true, y_pred, zero_division=0)
        }
    
    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        name: str
    ):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Dead (0)', 'Alive (1)']
        )
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        plt.title(f'{name} - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f"{name}_confusion_matrix.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved confusion matrix: {save_path}")
    
    def save_results_summary(self):
        """Save summary of all results to CSV."""
        df = pd.DataFrame(self.results)
        csv_path = os.path.join(self.output_dir, "final_results.csv")
        df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"\nFinal results saved to: {csv_path}")
        return df


def load_epoch_config(config_path: str) -> Dict:
    """
    Load epoch configuration from CSV or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary mapping dataset names to epochs
    """
    if config_path.endswith('.csv'):
        df = pd.read_csv(config_path, index_col=0)
        return df['best_epoch'].to_dict()
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError("Config file must be CSV or JSON")


def process_input_directory(
    input_dir: str,
    trainer: FinalTrainer,
    epochs_config: Dict,
    default_epochs: int,
    batch_size: int
):
    """
    Process all matched pickle files with specified epochs.
    
    Args:
        input_dir: Directory containing pickle files
        trainer: FinalTrainer instance
        epochs_config: Dictionary mapping names to optimal epochs
        default_epochs: Default epochs if not in config
        batch_size: Batch size for training
    """
    # Find all dead and alive bank files
    dead_files = sorted(glob(os.path.join(input_dir, "dead_bank_*.pickle")))
    alive_files = sorted(glob(os.path.join(input_dir, "alive_bank_*.pickle")))
    
    # Create mappings
    dead_map = {DataProcessor.extract_key(f): f for f in dead_files}
    alive_map = {DataProcessor.extract_key(f): f for f in alive_files}
    common_keys = sorted(set(dead_map.keys()) & set(alive_map.keys()))
    
    print(f"\nFound {len(common_keys)} matched dataset pairs:")
    for key in common_keys:
        epochs = epochs_config.get(key, default_epochs)
        print(f"  - {key}: {epochs} epochs")
    
    # Process each pair
    for name in common_keys:
        dead_file = dead_map[name]
        alive_file = alive_map[name]
        
        # Get epochs for this dataset
        epochs = epochs_config.get(name, default_epochs)
        
        # Load data
        dead_dict = DataProcessor.load_pickle(dead_file)
        alive_dict = DataProcessor.load_pickle(alive_file)
        
        # Train and evaluate
        trainer.train_and_evaluate(
            name, dead_dict, alive_dict, epochs, batch_size
        )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Final training and testing phase for bank failure prediction'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing matched pickle files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./final_results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--epochs-config',
        type=str,
        default=None,
        help='Path to CSV/JSON with optimal epochs per dataset'
    )
    parser.add_argument(
        '--default-epochs',
        type=int,
        default=2500,
        help='Default epochs if not specified in config'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for training'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU device ID to use (None for CPU)'
    )
    
    args = parser.parse_args()
    
    # Set GPU if specified
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"Using GPU: {args.gpu}")
    else:
        print("Using CPU")
    
    # Load epoch configuration
    epochs_config = {}
    if args.epochs_config and os.path.exists(args.epochs_config):
        print(f"Loading epoch configuration from: {args.epochs_config}")
        epochs_config = load_epoch_config(args.epochs_config)
    else:
        print(f"Using default epochs: {args.default_epochs}")
    
    # Initialize trainer
    trainer = FinalTrainer(args.output_dir)
    
    # Process all datasets
    process_input_directory(
        args.input_dir,
        trainer,
        epochs_config,
        args.default_epochs,
        args.batch_size
    )
    
    # Save results summary
    results_df = trainer.save_results_summary()
    
    print("\n" + "="*60)
    print("FINAL TRAINING AND TESTING COMPLETE")
    print("="*60)
    print("\nFinal Results:")
    print(results_df.to_string(index=False))
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()