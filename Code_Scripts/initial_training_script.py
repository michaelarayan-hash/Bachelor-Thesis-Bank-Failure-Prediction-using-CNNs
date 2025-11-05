"""
Initial Training Phase for Bank Failure Prediction

This script performs initial training and plots training/validation accuracies
for each variable set to identify optimal epochs.
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from typing import Dict, Tuple
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split


class CNNModel:
    """Handles CNN model creation and training."""
    
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
        val_size: float = 0.5,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert dictionaries to image arrays and split into train/val/test.
        
        Args:
            dead_dict: Dictionary of dead bank arrays
            alive_dict: Dictionary of alive bank arrays
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
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
        
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
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


class Trainer:
    """Handles model training and evaluation."""
    
    def __init__(self, output_dir: str):
        """
        Initialize trainer.
        
        Args:
            output_dir: Directory to save plots and results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
    
    def train_model(
        self,
        name: str,
        dead_dict: Dict,
        alive_dict: Dict,
        epochs: int = 2500,
        batch_size: int = 64
    ):
        """
        Train model and save plots.
        
        Args:
            name: Name of the variable set
            dead_dict: Dictionary of dead bank arrays
            alive_dict: Dictionary of alive bank arrays
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        print(f"\n{'='*60}")
        print(f"Training model for: {name}")
        print(f"{'='*60}")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = DataProcessor.prepare_data(
            dead_dict, alive_dict
        )
        
        print(f"Data shapes:")
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Build model
        input_shape = X_train.shape[1:]
        model = CNNModel.build_cnn(input_shape, kernel_dim=3)
        
        print(f"\nModel Summary:")
        model.summary()
        
        # Train model
        print(f"\nTraining for {epochs} epochs...")
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # Evaluate on test set
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n{name} Results:")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        
        # Save results
        self.results.append({
            "Name": name,
            "Test Accuracy": round(test_acc, 4),
            "Test Loss": round(test_loss, 4)
        })
        
        # Plot and save figures
        self._plot_accuracy(history, name)
        self._plot_loss(history, name)
        
        # Find best epoch (highest validation accuracy)
        best_epoch = np.argmax(history.history['val_accuracy']) + 1
        best_val_acc = np.max(history.history['val_accuracy'])
        print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        
        return best_epoch, best_val_acc
    
    def _plot_accuracy(self, history, name: str):
        """Plot and save accuracy curves."""
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        epochs = range(1, len(train_acc) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'{name} - Accuracy', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f"{name}_accuracy.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved accuracy plot: {save_path}")
    
    def _plot_loss(self, history, name: str):
        """Plot and save loss curves."""
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss, 'bo-', label='Training Loss', markersize=3, linewidth=1.5)
        plt.plot(epochs, val_loss, 'b-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'{name} - Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f"{name}_loss.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved loss plot: {save_path}")
    
    def save_results_summary(self):
        """Save summary of all results to CSV."""
        df = pd.DataFrame(self.results)
        csv_path = os.path.join(self.output_dir, "results_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults summary saved to: {csv_path}")
        return df


def process_input_directory(input_dir: str, trainer: Trainer, epochs: int, batch_size: int):
    """
    Process all matched pickle files in input directory.
    
    Args:
        input_dir: Directory containing pickle files
        trainer: Trainer instance
        epochs: Number of training epochs
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
        print(f"  - {key}")
    
    best_epochs = {}
    
    # Process each pair
    for name in common_keys:
        dead_file = dead_map[name]
        alive_file = alive_map[name]
        
        # Load data
        dead_dict = DataProcessor.load_pickle(dead_file)
        alive_dict = DataProcessor.load_pickle(alive_file)
        
        # Train model
        best_epoch, best_val_acc = trainer.train_model(
            name, dead_dict, alive_dict, epochs, batch_size
        )
        
        best_epochs[name] = {
            'best_epoch': best_epoch,
            'best_val_accuracy': best_val_acc
        }
    
    # Save best epochs information
    best_epochs_df = pd.DataFrame.from_dict(best_epochs, orient='index')
    best_epochs_path = os.path.join(trainer.output_dir, "best_epochs.csv")
    best_epochs_df.to_csv(best_epochs_path)
    print(f"\nBest epochs saved to: {best_epochs_path}")
    
    return best_epochs


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Initial training phase for bank failure prediction'
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
        default='./plots',
        help='Directory to save plots and results'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=2500,
        help='Number of training epochs'
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
    
    # Initialize trainer
    trainer = Trainer(args.output_dir)
    
    # Process all datasets
    best_epochs = process_input_directory(
        args.input_dir,
        trainer,
        args.epochs,
        args.batch_size
    )
    
    # Save results summary
    results_df = trainer.save_results_summary()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nFinal Results:")
    print(results_df.to_string(index=False))
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()