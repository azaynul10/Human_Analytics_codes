# audio_fall_integrated.py
# Main script integrating modular components for fall detection

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoFeatureExtractor, 
    AutoModelForAudioClassification, 
    WavLMForSequenceClassification,
    get_linear_schedule_with_warmup
)
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") # Use Agg backend for non-interactive plotting
import seaborn as sns
import shutil
import pickle
import warnings
import joblib
import json
import multiprocessing
from datetime import datetime
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F

# --- Scikit-learn Imports ---
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_recall_curve, accuracy_score, precision_score, 
                           recall_score, f1_score, roc_auc_score, log_loss, brier_score_loss,
                           roc_curve, auc, average_precision_score, matthews_corrcoef, balanced_accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
from collections import Counter
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# --- LightGBM Import ---
import lightgbm as lgb

# --- PyTorch & Transformers Imports ---
# Note: Tensorflow/Keras imports removed as AST (PyTorch) replaces the CNN

# --- Custom Module Imports ---
# Assuming models.py, features.py, augmentations.py, error_analysis.py, optimization_utils.py 
# are in the same directory or Python path.
try:
    from models import (load_ast_model_for_finetuning, train_ast_model, predict_with_ast,
                       AudioDataset)  # Added AudioDataset import
    from features import (extract_combined_features_for_ml, 
                          create_spectrogram_for_dl, SR, DURATION, N_MELS)
    from augmentations import (get_dl_augmentations, generate_augmented_audio_for_ml, 
                               load_rir_files)
    from error_analysis import analyze_and_save_errors
    from optimization_utils import (optimize_ensemble_weights, 
                                  plot_precision_recall_vs_threshold, 
                                  find_optimal_threshold)
    # Import the 2025 research-based feature extractor which includes audio enhancement
    from enhanced_2025 import RobustFeatureExtractor2025
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure all required modules are accessible:")
    print("  - models.py")
    print("  - features.py")
    print("  - augmentations.py")
    print("  - error_analysis.py")
    print("  - optimization_utils.py")
    print("  - enhanced_2025.py (contains audio enhancement and feature extraction)")
    exit()

# Add logging configuration
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_panns_model(model_checkpoint="microsoft/wavlm-base", num_labels=2):
    """Load the PANNs model."""
    try:
        print(f"\nLoading PANNs model from {model_checkpoint}...")
        # Load model with correct configuration
        model = WavLMForSequenceClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_labels,
            output_hidden_states=True,
            ignore_mismatched_sizes=True  # Handle potential size mismatches
        )
        
        # Load feature extractor
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_checkpoint,
            sampling_rate=16000,  # Ensure consistent sampling rate
            padding=True,
            do_normalize=True
        )
        
        print("PANNs model loaded successfully")
        return model, feature_extractor
        
    except Exception as e:
        print(f"Error loading PANNs model: {e}")
        return None, None

class AudioDataset(Dataset):
    """Dataset class for audio files."""
    def __init__(self, file_paths, labels, feature_extractor, apply_mixup=False, mixup_alpha=0.2):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.apply_mixup = apply_mixup
        self.mixup_alpha = mixup_alpha
        self.max_length = 80000  # 5 seconds at 16kHz
        self.sampling_rate = 16000  # Consistent sampling rate
        
    def __len__(self):
        return len(self.file_paths)
        
    def process_audio(self, audio_path):
        """Process audio file with consistent parameters."""
        try:
            # Load audio with consistent sampling rate
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Pad or truncate to max_length
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            else:
                audio = np.pad(audio, (0, max(0, self.max_length - len(audio))))
            
            return audio
            
        except Exception as e:
            print(f"Error processing audio file {audio_path}: {e}")
            return np.zeros(self.max_length)
        
    def __getitem__(self, idx):
        try:
            # Process audio
            audio = self.process_audio(self.file_paths[idx])
            
            # Extract features with consistent parameters
            inputs = self.feature_extractor(
                audio,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                do_normalize=True
            )
            
            # Get label
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            
            # Apply mixup if enabled
            if self.apply_mixup and self.mixup_alpha > 0:
                # Get random sample
                rand_idx = np.random.randint(0, len(self.file_paths))
                rand_audio = self.process_audio(self.file_paths[rand_idx])
                
                # Extract features for random audio
                rand_inputs = self.feature_extractor(
                    rand_audio,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    do_normalize=True
                )
                
                rand_label = torch.tensor(self.labels[rand_idx], dtype=torch.long)
                
                # Generate mixup weights
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                
                # Mix inputs
                inputs = {k: lam * v + (1 - lam) * rand_inputs[k] for k, v in inputs.items()}
                
                # For labels, we'll use the original label if lam > 0.5, otherwise use the random label
                label = label if lam > 0.5 else rand_label
            
            return inputs, label
            
        except Exception as e:
            print(f"Error in __getitem__ for {self.file_paths[idx]}: {e}")
            # Return a default/empty sample in case of error
            empty_audio = np.zeros(self.max_length)
            empty_inputs = self.feature_extractor(
                empty_audio,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                do_normalize=True
            )
            
            return empty_inputs, torch.tensor(0, dtype=torch.long)

def collate_fn(batch):
    """Custom collate function to handle tensor shapes."""
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    batched_inputs = {}
    for key in inputs[0].keys():
        # Use torch.cat for input tensors to concatenate along batch dimension
        batched_inputs[key] = torch.cat([x[key] for x in inputs], dim=0)
    
    # Use torch.stack for labels
    batched_labels = torch.stack(labels)
    
    return batched_inputs, batched_labels

def train_panns_model(
    model,
    feature_extractor,
    train_files,
    train_labels,
    val_files,
    val_labels,
    epochs=10,
    batch_size=8,
    learning_rate=5e-5,
    output_dir="results/panns_finetuned",
    mixup_alpha=0.2,
    weight_decay=0.01,
    warmup_steps=100,
    gradient_accumulation_steps=1,
    class_weights=None
):
    """Train the PANNs model."""
    try:
        # Create datasets
        train_dataset = AudioDataset(train_files, train_labels, feature_extractor, apply_mixup=True, mixup_alpha=mixup_alpha)
        val_dataset = AudioDataset(val_files, val_labels, feature_extractor, apply_mixup=False)
        
        # Create data loaders with custom collate function
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn
        )
        
        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Set up optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        num_training_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Set up loss function
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(**inputs)
                loss = criterion(outputs.logits, labels)
                
                # Backward pass
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # Move inputs to device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    labels = labels.to(device)
                    
                    # Forward pass
                    outputs = model(**inputs)
                    loss = criterion(outputs.logits, labels)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            train_loss = total_loss / len(train_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(output_dir)
                print(f"Saved best model to {output_dir}")
        
        return model, train_losses, val_losses
        
    except Exception as e:
        print(f"Error in train_panns_model: {e}")
        import traceback
        traceback.print_exc()
        return None, [], []

def predict_with_panns(model, feature_extractor, X):
    """Make predictions using the PANNs model."""
    try:
        if model is None:
            print("Error: Model is None")
            return None
            
        model.eval()
        predictions = []
        max_length = 80000  # 5 seconds at 16kHz
        
        with torch.no_grad():
            for audio_path in X:
                try:
                    # Load and preprocess audio
                    audio, sr = librosa.load(audio_path, sr=16000)
                    
                    # Ensure audio is 1D array
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                    
                    # Pad or truncate to max_length
                    if len(audio) > max_length:
                        audio = audio[:max_length]
                    else:
                        audio = np.pad(audio, (0, max(0, max_length - len(audio))))
                    
                    # Extract features
                    inputs = feature_extractor(
                        audio,
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        do_normalize=True
                    )
                    
                    # Move inputs to device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Get predictions
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    predictions.append(probs.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error processing file {audio_path}: {e}")
                    predictions.append(np.array([[0.5, 0.5]]))  # Default prediction
        
        return np.vstack(predictions)
        
    except Exception as e:
        print(f"Error in predict_with_panns: {e}")
        return None

class FallDetectionVisualizer:
    """Enhanced visualization suite for fall detection system."""
    
    def __init__(self, figsize=(12, 8), style='seaborn-v0_8-whitegrid'):
        """Initialize the visualizer with specified style and figure size."""
        self.figsize = figsize
        self.style = style
        plt.style.use(style)
        
    def plot_training_progress(self, train_scores, val_scores, title="Training Progress", xlabel="Iteration", ylabel="Score"):
        """Plot training and validation scores over time."""
        try:
            # Check if we have any scores to plot
            if not train_scores or not val_scores:
                print(f"Warning: No scores to plot for {title}")
                return
                
            plt.figure(figsize=self.figsize)
            plt.plot(train_scores, label='Training', color='blue', alpha=0.7)
            plt.plot(val_scores, label='Validation', color='red', alpha=0.7)
            
            plt.title(title, fontsize=14, pad=20)
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Add best validation score annotation if we have scores
            if len(val_scores) > 0:
                best_val_idx = np.argmin(val_scores) if 'loss' in ylabel.lower() else np.argmax(val_scores)
                best_val_score = val_scores[best_val_idx]
                plt.annotate(f'Best Val: {best_val_score:.4f}',
                            xy=(best_val_idx, best_val_score),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # Save the plot
            os.makedirs(RESULTS_DIR, exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, f'{title.lower().replace(" ", "_")}.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Warning: Failed to create training progress plot: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_feature_importance_analysis(self, feature_names, importance_scores, model_names, title="Feature Importance Analysis"):
        """Plot feature importance analysis."""
        try:
            plt.figure(figsize=self.figsize)
            
            # Create bar plot
            x = np.arange(len(feature_names))
            width = 0.8 / len(model_names)
            
            for i, model_name in enumerate(model_names):
                plt.bar(x + i * width, importance_scores[i], width, label=model_name)
            
            plt.title(title, fontsize=14, pad=20)
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Importance Score', fontsize=12)
            plt.xticks(x + width * (len(model_names) - 1) / 2, feature_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            os.makedirs(RESULTS_DIR, exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Warning: Failed to create feature importance plot: {e}")
    
    def plot_model_architecture_diagram(self, model_components, title="Model Architecture"):
        """Create a detailed model architecture diagram."""
        try:
            plt.figure(figsize=self.figsize)
            
            # Calculate positions
            num_components = len(model_components)
            y_positions = np.linspace(0, 1, num_components)
            
            # Draw components
            for i, (name, details) in enumerate(model_components.items()):
                # Draw component box
                plt.gca().add_patch(plt.Rectangle(
                    (0.1, y_positions[i] - 0.1),
                    0.8, 0.15,
                    facecolor='lightblue',
                    edgecolor='navy',
                    alpha=0.6
                ))
                
                # Add component name
                plt.text(0.5, y_positions[i],
                        name,
                        ha='center', va='center',
                        fontsize=10, fontweight='bold')
                
                # Add details
                plt.text(0.5, y_positions[i] - 0.05,
                        details,
                        ha='center', va='center',
                        fontsize=8)
                
                # Draw connections
                if i < num_components - 1:
                    plt.arrow(0.5, y_positions[i] - 0.1,
                             0, y_positions[i+1] - y_positions[i] + 0.1,
                             head_width=0.02, head_length=0.02,
                             fc='navy', ec='navy')
            
            plt.title(title, pad=20)
            plt.axis('off')
            plt.tight_layout()
            
            # Ensure directory exists
            os.makedirs(os.path.join('results', 'visualizations'), exist_ok=True)
            # Save the diagram
            plt.savefig(os.path.join('results', 'visualizations', 'model_architecture.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Failed to create model architecture diagram: {e}")
            plt.close()
    
    def plot_audio_spectrogram_analysis(self, audio_data, predictions=None, attention=None, title="Audio Spectrogram Analysis"):
        """Plot audio spectrogram analysis with predictions."""
        try:
            plt.figure(figsize=self.figsize)
            
            # Load and plot spectrogram
            audio, sr = librosa.load(audio_data, sr=None)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            plt.imshow(mel_spec_db, aspect='auto', origin='lower')
            plt.colorbar(format='%+2.0f dB')
            
            if predictions is not None:
                plt.title(f"{title}\nPrediction: {predictions[0]:.2f}", fontsize=14, pad=20)
            else:
                plt.title(title, fontsize=14, pad=20)
                
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            
            # Save the plot
            os.makedirs(RESULTS_DIR, exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, 'spectrogram_analysis.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Warning: Failed to create spectrogram analysis: {e}")
    
    def plot_comprehensive_performance_metrics(self, y_true, y_prob, class_names, title="Performance Analysis"):
        """Plot comprehensive performance metrics including ROC, PR curves, and metrics."""
        try:
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Calculate PR curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
            fig.suptitle(title, fontsize=16)
            
            # Plot ROC curve
            ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('Receiver Operating Characteristic')
            ax1.legend(loc="lower right")
            
            # Plot PR curve
            ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall Curve')
            ax2.legend(loc="lower left")
            
            # Calculate and plot metrics
            thresholds = np.linspace(0, 1, 100)
            metrics = []
            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                metrics.append([threshold, precision, recall, f1])
            
            metrics = np.array(metrics)
            
            # Plot metrics vs threshold
            ax3.plot(metrics[:, 0], metrics[:, 1], 'b-', label='Precision')
            ax3.plot(metrics[:, 0], metrics[:, 2], 'g-', label='Recall')
            ax3.plot(metrics[:, 0], metrics[:, 3], 'r-', label='F1 Score')
            ax3.set_xlabel('Threshold')
            ax3.set_ylabel('Score')
            ax3.set_title('Metrics vs Threshold')
            ax3.legend()
            ax3.grid(True)
            
            # Plot distribution of predictions
            ax4.hist(y_prob[y_true == 0], bins=50, alpha=0.5, label=class_names[0])
            ax4.hist(y_prob[y_true == 1], bins=50, alpha=0.5, label=class_names[1])
            ax4.set_xlabel('Predicted Probability')
            ax4.set_ylabel('Count')
            ax4.set_title('Prediction Distribution')
            ax4.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'performance_metrics.png'))
            plt.close()
            
        except Exception as e:
            print(f"Warning: Failed to create performance metrics plot: {e}")
            import traceback; traceback.print_exc()
    
    def plot_research_grade_confusion_matrix(self, y_true, y_pred, class_names, title="Confusion Matrix"):
        """Create a research-grade confusion matrix visualization."""
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create figure
            plt.figure(figsize=self.figsize)
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names,
                       yticklabels=class_names)
            
            # Add labels and title
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(title)
            
            # Calculate metrics
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Add metrics text
            metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}'
            plt.text(1.5, 0.5, metrics_text, fontsize=10, va='center')
            
            # Save the plot
            os.makedirs('results/visualizations', exist_ok=True)
            plt.savefig(os.path.join('results', 'visualizations', 'confusion_matrix.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Failed to create confusion matrix: {e}")
            plt.close()
    
    def plot_workflow_flowchart(self, workflow_steps, title="Fall Detection System Workflow"):
        """Plot workflow flowchart for the fall detection system."""
        try:
            plt.figure(figsize=self.figsize)
            
            # Set up the plot
            plt.axis('off')
            plt.title(title, fontsize=14, pad=20)
            
            # Calculate positions
            n_sections = len(workflow_steps)
            section_height = 0.8 / n_sections
            y_positions = np.linspace(0.9, 0.1, n_sections)
            
            # Plot each section
            for i, (section_name, steps) in enumerate(workflow_steps.items()):
                # Plot section box
                plt.text(0.1, y_positions[i], section_name, 
                        fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='lightblue', alpha=0.5, boxstyle='round,pad=0.5'))
                
                # Plot steps
                for j, step in enumerate(steps):
                    step_y = y_positions[i] - (j + 1) * 0.05
                    plt.text(0.2, step_y, f"• {step}", fontsize=10)
                    
                    # Add connecting lines
                    if j < len(steps) - 1:
                        plt.plot([0.25, 0.25], [step_y, step_y - 0.05], 
                                color='gray', linestyle='-', alpha=0.5)
                
                # Add connecting lines between sections
                if i < n_sections - 1:
                    plt.plot([0.15, 0.15], 
                            [y_positions[i] - (len(steps) + 1) * 0.05, y_positions[i + 1]],
                            color='gray', linestyle='-', alpha=0.5)
            
            # Save the plot
            os.makedirs(RESULTS_DIR, exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, 'workflow_flowchart.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Warning: Failed to create workflow flowchart: {e}")
            import traceback
            traceback.print_exc()

# --- Configuration --- 
# User Action Required: Set these paths and parameters

# Data Directories
BASE_DATA_DIR = r"C:\Users\T2430477\Downloads\archive" # Base directory containing the SAFE dataset structure
DATA_DIR = os.path.join(BASE_DATA_DIR, "audio_clips") # Directory with Fold 1-10 .wav files
ESC50_DIR = r"C:\Users\T2430477\Downloads\ESC-50-master\ESC-50-master"  # Base directory containing audio and meta folders
AUGMENTED_DATA_DIR = os.path.join(BASE_DATA_DIR, "augmented_audio_ml") # Dir to store augmented audio for ML
CACHE_DIR = os.path.join(BASE_DATA_DIR, "cache") # Directory for caching processed features
RIR_DATASET_DIR = r"C:\Users\T2430477\Downloads\air_database_release_1_4\AIR_1_4\AIR_wav_files" # Directory containing RIR .wav files
FALL_AUDIO_DIR = r"C:\Users\T2430477\Downloads\raw\raw\normal"  # Directory containing all fall audio data

# Model & Output Directories
RESULTS_DIR = "results"
AST_OUTPUT_DIR = os.path.join(RESULTS_DIR, "ast_finetuned")
ML_MODEL_DIR = os.path.join(RESULTS_DIR, "ml_models")
ERROR_ANALYSIS_DIR = os.path.join(RESULTS_DIR, "error_analysis")

# AST Model Configuration
AST_MODEL_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"

# Training Parameters
TEST_FOLD = 10 # Which fold to use as the final test set
AST_EPOCHS = 50  # Increased from 35 to 50 for better convergence
AST_BATCH_SIZE = 8  # Reduced from 16 for better stability
AST_LEARNING_RATE = 1e-5  # Reduced from 1.5e-5 for more stable training
AST_WEIGHT_DECAY = 5e-4  # Increased from 2e-4 for stronger regularization
ML_BATCH_SIZE = 32 # Batch size for ML models
SEED = 42

# Enhanced Augmentation Parameters
NUM_ML_AUGMENTATIONS = {
    'fall': 20,  # Increased from 15 to 20 for more fall samples
    'non_fall': 5
}
APPLY_AUGMENTATION_ML = True
APPLY_AUGMENTATION_DL = True
AUGMENTATION_PROBABILITY = 0.9  # Increased from 0.8 to 0.9

# Augmentation Parameters for Different Sound Types
AUGMENTATION_CONFIG = {
    'fall': {
        'time_stretch': (0.7, 1.3),  # More aggressive time stretching
        'pitch_shift': (-3, 3),      # More aggressive pitch shifting
        'noise_level': (0.01, 0.1),  # Increased noise levels
        'rir_probability': 0.8,      # Increased room simulation
        'gain_range': (-6, 6)        # More aggressive gain variation
    },
    'non_fall': {
        'time_stretch': (0.7, 1.3),  # Same as fall
        'pitch_shift': (-3, 3),      # Same as fall
        'noise_level': (0.01, 0.1),  # Same as fall
        'rir_probability': 0.8,      # Same as fall
        'gain_range': (-6, 6)        # Same as fall
    }
}

# Class Weighting Configuration
FALL_CLASS_WEIGHT = 3.0  # Increased from 2.0 to 3.0 for better fall detection
NORMAL_CLASS_WEIGHT = 1.0  # Weight for normal class (class 0)

# Ensemble Configuration
ENSEMBLE_WEIGHTS = {
    'ast': 0.4,  # Increased from 0.3
    'lgb': 0.3,  # Kept at 0.3
    'rf': 0.3    # Decreased from 0.4
}
OPTIMIZE_WEIGHTS = True
OPTIMIZE_THRESHOLD_TARGET = "f1"  # Keep as "f1" for balanced performance
MIN_RECALL_CONSTRAINT = 0.85  # Increased from 0.80 to 0.85
MAX_FALSE_POSITIVE_RATE = 0.12  # Decreased from 0.15 to 0.12

# ESC-50 Category Filtering
RELEVANT_ESC50_CATEGORIES = {
    "coughing": 1.0,
    "sneezing": 1.0,
    "door_knock": 1.0,
    "glass_breaking": 1.0,
    "footsteps": 1.0,
    "door_slam": 1.0,
    "chair_moving": 1.0,
    "object_drop": 0.8,
    "phone_ringing": 0.8,
    "water_running": 0.8,
    "microwave": 0.8,
    "vacuum_cleaner": 0.8
}

# Early Stopping
EARLY_STOPPING_PATIENCE = 10  # Increased from 8 to 10
EARLY_STOPPING_DELTA = 0.0005  # Decreased from 0.001 for finer convergence

# Learning Rate Schedule
LR_SCHEDULER_FACTOR = 0.7  # Changed from 0.5 for gentler reduction
LR_SCHEDULER_PATIENCE = 4  # Increased from 3 to 4
LR_SCHEDULER_MIN_LR = 5e-7  # Decreased from 1e-6 for better convergence

# AST Model Training Parameters
ast_params = {
    'epochs': 50,  # Increased from 20 to 50
    'batch_size': 8,  # Kept at 8
    'learning_rate': 5e-5,  # Reduced from 1e-4
    'mixup_alpha': 0.5,  # Increased from 0.4
    'fall_class_weight': 3.0,  # Adjusted from 5.0
    'min_recall_threshold': 0.85,  # Increased from 0.80
    'patience': 10,  # Increased from 8
    'lr_scheduler_patience': 5,  # Increased from 4
    'lr_scheduler_factor': 0.7,  # Kept at 0.7
    'min_lr': 5e-7,  # Decreased from 1e-5
    'focal_loss_alpha': 0.8,  # Increased from 0.75
    'focal_loss_gamma': 3.0,  # Increased from 2.5
    'contrastive_weight': 0.25,  # Increased from 0.2
    'label_smoothing': 0.15  # Increased from 0.1
}

# --- Setup & Initialization ---

# Set seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Create output directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(AST_OUTPUT_DIR, exist_ok=True)
os.makedirs(ML_MODEL_DIR, exist_ok=True)
os.makedirs(ERROR_ANALYSIS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
if APPLY_AUGMENTATION_ML:
    os.makedirs(AUGMENTED_DATA_DIR, exist_ok=True)

# Configure CPU cores (Optional, for parallel processing in scikit-learn/joblib)
try:
    n_cores = multiprocessing.cpu_count()
    n_cores = max(1, int(n_cores * 0.75))
    joblib.parallel.DEFAULT_N_JOBS = n_cores
    print(f"Configured joblib to use up to {n_cores} CPU cores")
except Exception as e:
    print(f"Warning: Could not configure CPU cores for joblib: {e}")

# GPU Configuration (PyTorch)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")

# Initialize visualizer
viz = FallDetectionVisualizer()

# Constants
SR = 16000  # Sample rate
MAX_LENGTH = 16000 * 5  # 5 seconds
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
EPOCHS = 15

# --- Utility Functions ---

def load_audio_file(file_path, target_sr=SR):
    """Load and preprocess audio file."""
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=target_sr)
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        return audio, sr
        
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None

def parse_filename(filename):
    """Parse the structured filename format: AA-BBB-CC-DDD-FF.wav"""
    try:
        parts = os.path.splitext(filename)[0].split('-')
        if len(parts) != 5:
            return None
        fold, subject, env, seq, label = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        return fold, subject, env, seq, label
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return None

def prepare_and_cache_features(file_path, use_cache=True, fold=None):
    """Loads audio, extracts enhanced features, and caches the result."""
    if fold is not None:
        cache_filename = f"fold_{fold}_{os.path.basename(file_path)}.pkl"
    else:
        cache_filename = os.path.basename(file_path) + ".pkl"
    cache_path = os.path.join(CACHE_DIR, cache_filename)

    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            return cached_data
        except Exception as e:
            print(f"Cache read error for {cache_filename}: {e}")
            try:
                os.remove(cache_path)
            except Exception as del_e:
                print(f"Warning: Could not delete invalid cache file {cache_filename}: {del_e}")
            return None

    audio, sr = load_audio_file(file_path, target_sr=SR)
    if audio is None:
        print(f"Skipping {file_path}: Audio loading failed")
        return None

    # Extract enhanced features
    features = extract_enhanced_features(audio, sr)
    if features is None:
        print(f"Skipping {file_path}: Feature extraction failed")
        return None

    if use_cache:
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
        except Exception as e:
            print(f"Cache write error for {cache_filename}: {e}")
    
    return features

def extract_base_features(audio):
    """Extract base audio features."""
    try:
        # Calculate basic audio features
        mfccs = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=SR)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SR)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=SR)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)[0]
        
        # Calculate statistics for each feature
        features = []
        
        # MFCC statistics
        features.extend([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.max(mfccs, axis=1),
            np.min(mfccs, axis=1)
        ])
        
        # Spectral centroid statistics
        features.extend([
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.max(spectral_centroids),
            np.min(spectral_centroids)
        ])
        
        # Spectral rolloff statistics
        features.extend([
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.max(spectral_rolloff),
            np.min(spectral_rolloff)
        ])
        
        # Spectral bandwidth statistics
        features.extend([
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
            np.max(spectral_bandwidth),
            np.min(spectral_bandwidth)
        ])
        
        # Zero crossing rate statistics
        features.extend([
            np.mean(zero_crossing_rate),
            np.std(zero_crossing_rate),
            np.max(zero_crossing_rate),
            np.min(zero_crossing_rate)
        ])
        
        # Flatten and return features
        return np.concatenate([f.flatten() for f in features])
        
    except Exception as e:
        print(f"Error in extract_base_features: {e}")
        return None

def extract_enhanced_features(audio, sr):
    """Extract enhanced features including temporal, domain-specific, and selected features."""
    try:
        # Base features
        base_features = extract_base_features(audio)
        if base_features is None:
            return None
            
        # Temporal features
        temporal_features = extract_temporal_features(audio, sr)
        if temporal_features is None:
            return None
            
        # Domain-specific features
        domain_features = extract_domain_specific_features(audio, sr)
        if domain_features is None:
            return None
            
        # Combine all features
        all_features = np.concatenate([
            base_features,
            temporal_features,
            domain_features
        ])
        
        # For single samples, skip feature selection
        return all_features
        
    except Exception as e:
        print(f"Error in extract_enhanced_features: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_temporal_features(audio, sr):
    """Extract temporal features for fall detection."""
    try:
        # Short-time energy
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
        
        # Calculate temporal statistics
        temporal_stats = np.array([
            np.mean(energy), np.std(energy), np.max(energy), np.min(energy),
            np.mean(zcr), np.std(zcr), np.max(zcr), np.min(zcr),
            np.mean(centroid), np.std(centroid), np.max(centroid), np.min(centroid),
            np.mean(rolloff), np.std(rolloff), np.max(rolloff), np.min(rolloff)
        ])
        
        # Calculate temporal patterns
        energy_diff = np.diff(energy)
        zcr_diff = np.diff(zcr)
        centroid_diff = np.diff(centroid)
        
        pattern_stats = np.array([
            np.mean(energy_diff), np.std(energy_diff),
            np.mean(zcr_diff), np.std(zcr_diff),
            np.mean(centroid_diff), np.std(centroid_diff)
        ])
        
        # Ensure all arrays are 1D and have the same shape
        temporal_stats = temporal_stats.flatten()
        pattern_stats = pattern_stats.flatten()
        
        # Combine all temporal features
        temporal_features = np.concatenate([temporal_stats, pattern_stats])
        
        return temporal_features
        
    except Exception as e:
        print(f"Error in extract_temporal_features: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_domain_specific_features(audio, sr):
    """Extract domain-specific features for fall detection."""
    try:
        # Impact detection features
        impact_features = detect_impact(audio, sr)
        
        # Motion features
        motion_features = extract_motion_features(audio, sr)
        
        # Environmental features
        env_features = extract_environmental_features(audio, sr)
        
        # Combine domain-specific features
        domain_features = np.concatenate([
            impact_features,
            motion_features,
            env_features
        ])
        
        return domain_features
        
    except Exception as e:
        print(f"Error in extract_domain_specific_features: {e}")
        return None

def detect_impact(audio, sr):
    """Detect impact sounds in audio."""
    try:
        # Short-time energy
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Find potential impacts
        threshold = np.mean(energy) + 2 * np.std(energy)
        impact_indices = np.where(energy > threshold)[0]
        
        if len(impact_indices) > 0:
            # Impact features
            impact_features = np.array([
                len(impact_indices),  # Number of impacts
                np.mean(energy[impact_indices]),  # Mean impact energy
                np.max(energy[impact_indices]),   # Max impact energy
                np.std(energy[impact_indices]),   # Impact energy variation
                np.mean(np.diff(impact_indices))  # Mean time between impacts
            ])
        else:
            impact_features = np.zeros(5)
        
        return impact_features
        
    except Exception as e:
        print(f"Error in detect_impact: {e}")
        return np.zeros(5)

def extract_motion_features(audio, sr):
    """Extract motion-related features."""
    try:
        # Calculate spectral flux manually
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        # Compute spectrogram
        D = np.abs(librosa.stft(audio, n_fft=frame_length, hop_length=hop_length))
        
        # Calculate spectral flux
        flux = np.diff(D, axis=1)
        flux = np.sqrt(np.sum(flux**2, axis=0))
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
        
        # Motion features
        motion_features = np.array([
            np.mean(flux), np.std(flux), np.max(flux),
            np.mean(contrast), np.std(contrast), np.max(contrast)
        ])
        
        return motion_features.flatten()
        
    except Exception as e:
        print(f"Error in extract_motion_features: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros(6)  # Return zero array of correct size instead of None

def extract_environmental_features(audio, sr):
    """Extract environmental features from audio."""
    try:
        # Calculate signal-to-noise ratio
        signal_power = np.mean(audio ** 2)
        noise_power = np.var(audio)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Calculate spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        # Calculate spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        
        # Calculate spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        
        # Combine features
        features = np.concatenate([
            [snr],
            [np.mean(spectral_centroid)],
            [np.std(spectral_centroid)],
            [np.mean(spectral_bandwidth)],
            [np.std(spectral_bandwidth)],
            [np.mean(spectral_rolloff)],
            [np.std(spectral_rolloff)]
        ])
        
        return features
        
    except Exception as e:
        print(f"Error in extract_environmental_features: {e}")
        return None

def select_features(features, method='variance'):
    """Select most important features using various methods."""
    try:
        # Ensure features are 2D array
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            
        # For single samples, skip variance threshold and return all features
        if features.shape[0] == 1:
            return features.flatten()
            
        if method == 'variance':
            # Variance threshold with lower threshold
            selector = VarianceThreshold(threshold=0.001)  # Lowered threshold
            selected_features = selector.fit_transform(features)
            
        elif method == 'mutual_info':
            # Mutual information
            selector = SelectKBest(mutual_info_classif, k=100)
            selected_features = selector.fit_transform(features, np.zeros(features.shape[0]))
            
        elif method == 'l1':
            # L1-based selection
            selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear'))
            selected_features = selector.fit_transform(features, np.zeros(features.shape[0]))
            
        else:
            # Default to all features
            selected_features = features
            
        return selected_features.flatten()
        
    except Exception as e:
        print(f"Error in select_features: {e}")
        import traceback
        traceback.print_exc()
        return features.flatten()  # Return original features if selection fails

def load_dataset(data_dir, test_fold=TEST_FOLD):
    """Loads the dataset with data leakage prevention."""
    print(f"\nLoading dataset from: {data_dir}")
    all_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".wav")]
    print(f"Found {len(all_files)} .wav files in main dataset.")

    # Track files by subject to prevent leakage
    subject_files = {}
    data = []
    
    for filename in tqdm(all_files, desc="Parsing main dataset filenames"): 
        file_path = os.path.join(data_dir, filename)
        metadata = parse_filename(filename)
        if metadata:
            fold, subject, env, _, label = metadata
            if label in [1, 2]: # Valid labels
                mapped_label = label - 1 # 0 for normal, 1 for fall
                
                # Track files by subject
                if subject not in subject_files:
                    subject_files[subject] = []
                subject_files[subject].append({
                    "file_path": file_path,
                    "filename": filename,
                    "fold": fold,
                    "subject": subject,
                    "environment": env,
                    "label": mapped_label
                })
            else:
                print(f"Skipping {filename}: Invalid label {label}")
        else:
            print(f"Skipping {filename}: Invalid filename format")

    # Ensure no subject appears in multiple folds
    for subject, files in subject_files.items():
        folds = set(f["fold"] for f in files)
        if len(folds) > 1:
            print(f"Warning: Subject {subject} appears in multiple folds: {folds}")
            # Keep only files from the most common fold
            fold_counts = {}
            for f in files:
                fold_counts[f["fold"]] = fold_counts.get(f["fold"], 0) + 1
            most_common_fold = max(fold_counts.items(), key=lambda x: x[1])[0]
            files = [f for f in files if f["fold"] == most_common_fold]
        data.extend(files)

    # Add ESC-50 files with leakage prevention
    print(f"\nLoading ESC-50 dataset from: {ESC50_DIR}")
    esc50_files = [f for f in os.listdir(ESC50_DIR) if f.lower().endswith(".wav")]
    print(f"Found {len(esc50_files)} total .wav files in ESC-50 dataset.")

    # Randomly assign ESC-50 files to folds
    np.random.seed(42)  # For reproducibility
    esc50_folds = np.random.randint(1, 6, size=len(esc50_files))
    
    for filename, fold in tqdm(zip(esc50_files, esc50_folds), desc="Adding relevant ESC-50 files"):
        category = filename.split('-')[0].lower()
        if category in RELEVANT_ESC50_CATEGORIES:
            file_path = os.path.join(ESC50_DIR, filename)
            data.append({
                "file_path": file_path,
                "filename": filename,
                "fold": fold,
                "subject": -1,
                "environment": -1,
                "label": 0,
                "category": category,
                "confidence": RELEVANT_ESC50_CATEGORIES[category]
            })

    # Convert to DataFrame and verify no leakage
    df = pd.DataFrame(data)
    
    # Verify no file appears in multiple folds
    file_folds = df.groupby('file_path')['fold'].nunique()
    if (file_folds > 1).any():
        print("Warning: Some files appear in multiple folds!")
        print(file_folds[file_folds > 1])
    
    # Verify no subject appears in multiple folds
    subject_folds = df[df['subject'] != -1].groupby('subject')['fold'].nunique()
    if (subject_folds > 1).any():
        print("Warning: Some subjects appear in multiple folds!")
        print(subject_folds[subject_folds > 1])
    
    # Split into Train/Validation (Folds 1 to 9) and Test (Fold 10)
    train_val_df = df[df["fold"] != test_fold].reset_index(drop=True)
    test_df = df[df["fold"] == test_fold].reset_index(drop=True)

    print(f"\nTrain/Validation samples: {len(train_val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Further split Train/Validation into Train and Validation (e.g., use Fold 9 for validation)
    val_fold = test_fold - 1
    train_df = train_val_df[train_val_df["fold"] != val_fold].reset_index(drop=True)
    val_df = train_val_df[train_val_df["fold"] == val_fold].reset_index(drop=True)
    
    print(f"Final Training samples: {len(train_df)}")
    print(f"Final Validation samples: {len(val_df)}")
    print("\nTraining set class distribution:")
    print(train_df["label"].value_counts())
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print("\nClass distribution:")
    print(df["label"].value_counts())
    
    return train_df, val_df, test_df

class TemperatureScaling:
    """Temperature scaling calibration for model probabilities."""
    def __init__(self):
        self.temperature = torch.nn.Parameter(torch.ones(1))
    
    def fit(self, logits, labels):
        """Fit temperature scaling parameters."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=100)
        def eval():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = torch.nn.functional.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss
        optimizer.step(eval)
    
    def predict_proba(self, logits):
        """Predict calibrated probabilities."""
        with torch.no_grad():
            scaled_logits = logits / self.temperature
            return torch.softmax(scaled_logits, dim=1).detach().numpy()
    
    def __getstate__(self):
        """Make the class pickleable."""
        state = self.__dict__.copy()
        state['temperature'] = self.temperature.detach().numpy()
        return state
    
    def __setstate__(self, state):
        """Restore state from pickle."""
        self.__dict__.update(state)
        self.temperature = torch.nn.Parameter(torch.tensor(self.temperature))

def calibrate_predictions(model, X_train, y_train, X_val, y_val, method='isotonic'):
    """Calibrate model predictions using various methods."""
    try:
        if method == 'temperature':
            calibrator = TemperatureScaling()
            
            # Get model predictions
            if isinstance(model, lgb.Booster):
                # For LightGBM
                logits = model.predict(X_train, raw_score=True)
                logits = torch.tensor(logits).reshape(-1, 1)
                logits = torch.cat([-logits, logits], dim=1)
            elif isinstance(model, RandomForestClassifier):
                # For Random Forest
                probs = model.predict_proba(X_train)
                logits = torch.tensor(np.log(probs + 1e-10))
            else:
                # For AST model
                probs = predict_with_ast(model, None, X_train)
                if probs is None:
                    return None
                logits = torch.tensor(np.log(probs + 1e-10))
            
            # Convert labels to tensor
            labels = torch.tensor(y_train)
            
            # Fit calibrator
            calibrator.fit(logits, labels)
            return calibrator
            
        elif method == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            
            # Get model predictions
            if isinstance(model, lgb.Booster):
                probs = model.predict(X_train)
            elif isinstance(model, RandomForestClassifier):
                probs = model.predict_proba(X_train)[:, 1]
            else:
                # For AST model
                probs = predict_with_ast(model, None, X_train)
                if probs is None:
                    return None
                probs = probs[:, 1]
            
            # Fit calibrator
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(probs, y_train)
            return calibrator
            
        elif method == 'platt':
            from sklearn.linear_model import LogisticRegression
            
            # Get model predictions
            if isinstance(model, lgb.Booster):
                probs = model.predict(X_train)
            elif isinstance(model, RandomForestClassifier):
                probs = model.predict_proba(X_train)[:, 1]
            else:
                # For AST model
                probs = predict_with_ast(model, None, X_train)
                if probs is None:
                    return None
                probs = probs[:, 1]
            
            # Reshape for sklearn
            probs = probs.reshape(-1, 1)
            
            # Fit calibrator
            calibrator = LogisticRegression(solver='liblinear')
            calibrator.fit(probs, y_train)
            return calibrator
            
        else:
            print(f"Unknown calibration method: {method}")
            return None
            
    except Exception as e:
        print(f"Error in calibrate_predictions: {e}")
        import traceback
        traceback.print_exc()
        return None

def apply_calibration(calibrator, model, X, model_type='lgb'):
    """Apply calibration to model predictions."""
    try:
        if calibrator is None:
            return None
            
        if isinstance(calibrator, TemperatureScaling):
            if model_type == 'lgb':
                logits = model.predict(X, raw_score=True)
                logits = torch.tensor(logits).reshape(-1, 1)
                logits = torch.cat([-logits, logits], dim=1)
            elif model_type == 'rf':
                probs = model.predict_proba(X)
                logits = torch.tensor(np.log(probs + 1e-10))
            else:  # ast
                probs = predict_with_ast(model, None, X)
                if probs is None:
                    return None
                logits = torch.tensor(np.log(probs + 1e-10))
            return calibrator.predict_proba(logits)[:, 1].numpy()
            
        elif isinstance(calibrator, IsotonicRegression):
            if model_type == 'lgb':
                probs = model.predict(X)
            elif model_type == 'rf':
                probs = model.predict_proba(X)[:, 1]
            else:  # ast
                probs = predict_with_ast(model, None, X)
                if probs is None:
                    return None
                probs = probs[:, 1]
            return calibrator.predict(probs)
            
        elif isinstance(calibrator, LogisticRegression):
            if model_type == 'lgb':
                probs = model.predict(X)
            elif model_type == 'rf':
                probs = model.predict_proba(X)[:, 1]
            else:  # ast
                probs = predict_with_ast(model, None, X)
                if probs is None:
                    return None
                probs = probs[:, 1]
            return calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
            
        else:
            print(f"Unknown calibrator type: {type(calibrator)}")
            return None
            
    except Exception as e:
        print(f"Error in apply_calibration: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_ml_models(train_files, train_labels, val_files, val_labels, class_weights=None):
    """Train ML models with enhanced balancing."""
    try:
        print("\n=== Training ML Models with Enhanced Balancing ===\n")
        
        # Calculate current class distribution
        fall_count = sum(train_labels == 1)
        non_fall_count = sum(train_labels == 0)
        current_ratio = fall_count / (fall_count + non_fall_count)
        
        print(f"Current class distribution:")
        print(f"Fall samples: {fall_count}")
        print(f"Non-fall samples: {non_fall_count}")
        print(f"Current ratio: {current_ratio:.2f}")
        
        if class_weights is not None:
            print(f"\nClass weights:")
            print(f"Fall class weight: {class_weights[1]:.2f}")
            print(f"Non-fall class weight: {class_weights[0]:.2f}")
        
        print("\nExtracting features...")
        X_train, y_train = extract_features_for_ml(train_files, train_labels)
        if X_train is None or y_train is None:
            raise ValueError("Failed to extract features for training")
            
        X_val, y_val = extract_features_for_ml(val_files, val_labels)
        if X_val is None or y_val is None:
            raise ValueError("Failed to extract features for validation")
            
        # Train models
        models = {}
        calibrators = {}
        
        # Train Random Forest
        print("\nTraining Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight=class_weights,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        models['rf'] = rf_model
        
        # Train LightGBM
        print("\nTraining LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight=class_weights,
            random_state=42
        )
        lgb_model.fit(X_train, y_train)
        models['lgb'] = lgb_model
        
        # Calibrate models
        print("\nCalibrating models...")
        for model_name, model in models.items():
            # Get predictions
            if model_name == 'lgb':
                probs = model.predict_proba(X_val)[:, 1]
            else:
                probs = model.predict_proba(X_val)[:, 1]
                
            # Train calibrator
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(probs, y_val)
            calibrators[model_name] = calibrator
            
        return models, calibrators
        
    except Exception as e:
        print(f"Error in train_ml_models: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def predict_ml_models(lgb_model, rf_model, panns_model, scaler, file_paths):
    """Get predictions from all ML models."""
    try:
        print("\n=== Getting ML Model Predictions ===")
        
        # Extract features
        features = []
        valid_paths = []
        for file_path in tqdm(file_paths, desc="Extracting features"):
            feature = prepare_and_cache_features(file_path)
            if feature is not None:
                features.append(feature)
                valid_paths.append(file_path)
            else:
                print(f"Warning: Feature extraction failed for {file_path}")
        
        if not features:
            print("Error: No valid features extracted")
            return None, None, None, None
            
        X = np.array(features)
        X_scaled = scaler.transform(X)
        
        # Get LightGBM predictions
        print("\nGetting LightGBM predictions...")
        lgb_probs = lgb_model.predict(X_scaled)
        
        # Get Random Forest predictions
        print("\nGetting Random Forest predictions...")
        rf_probs = rf_model.predict_proba(X_scaled)[:, 1]
        
        # Get PANNs predictions
        print("\nGetting PANNs predictions...")
        panns_probs = predict_with_panns(panns_model, panns_feature_extractor, valid_paths)
        if panns_probs is None:
            print("Error: PANNs prediction failed")
            return None, None, None, None
            
        # Load ensemble weights
        with open(os.path.join(ML_MODEL_DIR, 'ensemble_config.json'), 'r') as f:
            ensemble_config = json.load(f)
        weights = ensemble_config['weights']
        
        # Calculate ensemble predictions
        ensemble_probs = (
            weights['lgb'] * lgb_probs +
            weights['rf'] * rf_probs +
            weights['panns'] * panns_probs
        )
        
        # Calculate uncertainties
        uncertainties = calculate_uncertainty([lgb_probs, rf_probs, panns_probs])
        
        return ensemble_probs, uncertainties, valid_paths, weights
        
    except Exception as e:
        print(f"Error in predict_ml_models: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

# --- AST Fine-tuning & Prediction ---
# Uses functions from models.py

# --- Ensemble Prediction ---
def post_process_prediction(ensemble_prob, audio_features):
    """Apply post-processing rules to reduce false positives."""
    # Rule 1: Check impact pattern
    if audio_features['impact_density'] > 0.3:  # Too many impacts for a fall
        ensemble_prob *= 0.5
    
    # Rule 2: Check decay rate
    if audio_features['impact_decay_mean'] < -0.1:  # Slow decay, likely not a fall
        ensemble_prob *= 0.7
    
    # Rule 3: Check temporal pattern
    if audio_features['impact_interval_std'] < 0.1:  # Regular intervals, likely footsteps
        ensemble_prob *= 0.6
        
    return ensemble_prob

def calculate_confidence_score(ensemble_prob, uncertainties):
    """Calculate confidence score based on ensemble probability and uncertainties.
    
    Args:
        ensemble_prob (float): Ensemble prediction probability
        uncertainties (dict): Dictionary of uncertainties for each model
        
    Returns:
        float: Confidence score between 0 and 1
    """
    # Calculate base confidence from ensemble probability
    # Higher confidence when probability is closer to 0 or 1
    prob_confidence = 1 - 2 * abs(ensemble_prob - 0.5)
    
    # Calculate uncertainty penalty
    # Higher uncertainty means lower confidence
    uncertainty_penalty = np.mean(list(uncertainties.values()))
    
    # Combine into final confidence score
    confidence_score = prob_confidence * (1 - uncertainty_penalty)
    
    return np.clip(confidence_score, 0, 1)

def analyze_temporal_pattern(probs, window_size=5):
    """Analyze temporal patterns in prediction probabilities.
    
    Args:
        probs (np.ndarray): Array of prediction probabilities
        window_size (int): Size of sliding window for analysis
        
    Returns:
        dict: Dictionary containing temporal pattern metrics
    """
    if len(probs) < window_size:
        return {
            'trend': 0,
            'volatility': 0,
            'consistency': 0,
            'pattern_score': 0
        }
    
    # Calculate trend (slope of linear regression)
    x = np.arange(len(probs))
    slope, _ = np.polyfit(x, probs, 1)
    
    # Calculate volatility (standard deviation of differences)
    diffs = np.diff(probs)
    volatility = np.std(diffs) if len(diffs) > 0 else 0
    
    # Calculate consistency (how often predictions change direction)
    direction_changes = np.sum(np.diff(np.signbit(diffs))) if len(diffs) > 0 else 0
    consistency = 1 - (direction_changes / (len(diffs) + 1e-10))
    
    # Calculate pattern score (combination of metrics)
    pattern_score = (
        0.4 * (1 - abs(slope)) +  # Prefer stable trends
        0.3 * (1 - volatility) +  # Prefer low volatility
        0.3 * consistency  # Prefer consistent predictions
    )
    
    return {
        'trend': slope,
        'volatility': volatility,
        'consistency': consistency,
        'pattern_score': pattern_score
    }

def detect_fall_pattern(probs, window_size=5, threshold=0.65):
    """Detect patterns typical of falls in prediction probabilities.
    
    Args:
        probs (np.ndarray): Array of prediction probabilities
        window_size (int): Size of sliding window for analysis
        threshold (float): Threshold for fall detection
        
    Returns:
        dict: Dictionary containing fall pattern metrics
    """
    if len(probs) < window_size:
        return {
            'is_fall_pattern': False,
            'confidence': 0,
            'pattern_type': 'unknown'
        }
    
    # Calculate moving average
    moving_avg = np.convolve(probs, np.ones(window_size)/window_size, mode='valid')
    
    # Detect sudden changes
    diffs = np.diff(moving_avg)
    sudden_changes = np.abs(diffs) > 0.3  # Threshold for sudden change
    
    # Analyze pattern characteristics
    pattern_metrics = {
        'sudden_rise': np.any((diffs > 0.3) & (moving_avg[:-1] < threshold)),
        'sustained_high': np.mean(moving_avg > threshold) > 0.7,
        'volatility': np.std(diffs) > 0.2
    }
    
    # Determine pattern type and confidence
    if pattern_metrics['sudden_rise'] and pattern_metrics['sustained_high']:
        pattern_type = 'sudden_fall'
        confidence = 0.9
    elif pattern_metrics['sustained_high']:
        pattern_type = 'gradual_fall'
        confidence = 0.7
    elif pattern_metrics['volatility']:
        pattern_type = 'uncertain'
        confidence = 0.5
    else:
        pattern_type = 'normal'
        confidence = 0.3
    
    return {
        'is_fall_pattern': pattern_type in ['sudden_fall', 'gradual_fall'],
        'confidence': confidence,
        'pattern_type': pattern_type
    }

def predict_ensemble(ast_probs, lgb_probs, rf_probs, weights):
    """Make ensemble predictions with uncertainty estimation."""
    try:
        # Ensure all probabilities are 2D arrays
        if ast_probs is not None:
            ast_probs = np.array(ast_probs).reshape(-1, 1)
        if lgb_probs is not None:
            lgb_probs = np.array(lgb_probs).reshape(-1, 1)
        if rf_probs is not None:
            rf_probs = np.array(rf_probs).reshape(-1, 1)
        
        # Initialize arrays for weighted probabilities
        weighted_probs = []
        uncertainties = []
        
        # Process each model's predictions
        if ast_probs is not None and 'ast' in weights:
            weighted_probs.append(ast_probs * weights['ast'])
            uncertainties.append(np.abs(ast_probs - 0.5) * weights['ast'])
            
        if lgb_probs is not None and 'lgb' in weights:
            weighted_probs.append(lgb_probs * weights['lgb'])
            uncertainties.append(np.abs(lgb_probs - 0.5) * weights['lgb'])
            
        if rf_probs is not None and 'rf' in weights:
            weighted_probs.append(rf_probs * weights['rf'])
            uncertainties.append(np.abs(rf_probs - 0.5) * weights['rf'])
        
        if not weighted_probs:
            raise ValueError("No valid model predictions available")
        
        # Combine predictions
        ensemble_prob = np.sum(weighted_probs, axis=0)
        ensemble_uncertainty = np.sum(uncertainties, axis=0)
        
        # Calculate confidence score
        confidence = calculate_confidence_score(ensemble_prob, ensemble_uncertainty)
        
        return ensemble_prob, confidence
        
    except Exception as e:
        print(f"Error in predict_ensemble: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def calculate_uncertainty(probs):
    """Calculate prediction uncertainty using entropy."""
    try:
        # Ensure probabilities are valid
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs))
        
        # Normalize entropy to [0, 1]
        max_entropy = -np.log2(0.5)  # Maximum entropy for binary classification
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
        
    except Exception as e:
        print(f"Error in calculate_uncertainty: {e}")
        return 0.5

def clear_cache_directory():
    """Clears the entire cache directory to force fresh feature extraction."""
    print(f"\nClearing cache directory: {CACHE_DIR}")
    try:
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR)
            print("Cache directory cleared successfully.")
        else:
            print("Cache directory does not exist. Creating it.")
            os.makedirs(CACHE_DIR)
    except Exception as e:
        print(f"Error clearing cache directory: {e}")
        return False
    return True

def sanitize_features(features, dtype=np.float32):
    """Sanitize features to ensure they are valid and of the correct type."""
    try:
        # Convert to numpy array if not already
        features = np.array(features, dtype=dtype)
        
        # Replace inf and -inf with large finite values
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Clip values to reasonable range
        features = np.clip(features, -1e10, 1e10)
        
        return features
        
    except Exception as e:
        print(f"Error in sanitize_features: {e}")
        return None

def extract_features_for_ml(file_paths, labels):
    """Extract features for ML models from audio files."""
    try:
        features = []
        valid_indices = []
        
        for idx, file_path in enumerate(file_paths):
            # Load and preprocess audio
            audio, sr = load_audio_file(file_path)
            if audio is None:
                print(f"Warning: Failed to load audio file {file_path}")
                continue
                
            # Extract features
            audio_features = extract_enhanced_features(audio, sr)
            if audio_features is None:
                print(f"Warning: Feature extraction failed for {file_path}")
                continue
                
            # Add temporal features
            temporal_features = extract_temporal_features(audio, sr)
            if temporal_features is not None:
                audio_features = np.concatenate([audio_features, temporal_features])
                
            # Add domain-specific features
            domain_features = extract_domain_specific_features(audio, sr)
            if domain_features is not None:
                audio_features = np.concatenate([audio_features, domain_features])
                
            # Add motion features
            motion_features = extract_motion_features(audio, sr)
            if motion_features is not None:
                audio_features = np.concatenate([audio_features, motion_features])
                
            # Add environmental features
            env_features = extract_environmental_features(audio, sr)
            if env_features is not None:
                audio_features = np.concatenate([audio_features, env_features])
                
            # Sanitize features
            audio_features = sanitize_features(audio_features)
            
            features.append(audio_features)
            valid_indices.append(idx)
            
        if not features:
            print("Error: No valid features extracted")
            return None, None
            
        X = np.array(features)
        y = labels[valid_indices]
        
        return X, y
        
    except Exception as e:
        print(f"Error in extract_features_for_ml: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def balance_dataset_with_esc50(df, target_ratio=0.7, min_confidence=0.6):
    """Enhanced dataset balancing using ESC-50 and class weights."""
    try:
        # Calculate current class distribution
        fall_count = sum(df['label'] == 1)
        non_fall_count = sum(df['label'] == 0)
        current_ratio = fall_count / (fall_count + non_fall_count)
        
        print(f"\nCurrent class distribution:")
        print(f"Fall samples: {fall_count}")
        print(f"Non-fall samples: {non_fall_count}")
        print(f"Current ratio: {current_ratio:.2f}")
        
        # Try to load ESC-50 dataset
        esc50_path = os.path.join(ESC50_DIR, 'meta', 'esc50.csv')
        print(f"\nLoading ESC-50 dataset from: {ESC50_DIR}")
        
        if not os.path.exists(esc50_path):
            print("\nESC-50 dataset not found. Using class weights only for balancing.")
            # Calculate class weights
            total_samples = len(df)
            fall_weight = total_samples / (2 * fall_count)
            non_fall_weight = total_samples / (2 * non_fall_count)
            
            class_weights = {
                0: non_fall_weight,
                1: fall_weight
            }
            
            print(f"\nClass weights:")
            print(f"Fall class weight: {fall_weight:.2f}")
            print(f"Non-fall class weight: {non_fall_weight:.2f}")
            
            return df, class_weights
        
        # Load ESC-50 dataset
        esc50_df = pd.read_csv(esc50_path)
        esc50_files = esc50_df['filename'].values
        esc50_folds = esc50_df['fold'].values
        
        # Count total .wav files
        wav_count = sum(1 for f in esc50_files if f.endswith('.wav'))
        print(f"Found {wav_count} total .wav files in ESC-50 dataset.")
        
        # Filter relevant ESC-50 categories with confidence threshold
        relevant_files = []
        for filename, fold in zip(esc50_files, esc50_folds):
            if not filename.endswith('.wav'):
                continue
                
            category = filename.split('-')[0].lower()
            if category in RELEVANT_ESC50_CATEGORIES:
                confidence = RELEVANT_ESC50_CATEGORIES[category]
                if confidence >= min_confidence:
                    # Use the correct audio file path
                    audio_path = os.path.join(ESC50_DIR, 'audio', filename)
                    if os.path.exists(audio_path):
                        relevant_files.append({
                            'file_path': audio_path,
                            'filename': filename,
                            'fold': fold,
                            'label': 0,
                            'confidence': confidence
                        })
        
        print(f"Adding relevant ESC-50 files: {len(relevant_files)}")
        
        # Calculate required additional non-fall samples
        target_fall_count = fall_count
        target_non_fall_count = int(target_fall_count * (1 - target_ratio) / target_ratio)
        additional_non_fall_needed = max(0, target_non_fall_count - non_fall_count)
        
        if additional_non_fall_needed > 0:
            # Sort ESC-50 files by confidence
            relevant_files.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Add ESC-50 files up to the target ratio
            added_count = 0
            for file_info in tqdm(relevant_files, desc="Adding ESC-50 files"):
                if added_count >= additional_non_fall_needed:
                    break
                    
                # Verify file doesn't exist in current dataset
                if not any(df['file_path'] == file_info['file_path']):
                    df = pd.concat([df, pd.DataFrame([file_info])], ignore_index=True)
                    added_count += 1
            
            print(f"\nAdded {added_count} ESC-50 files for balancing")
        
        # Calculate class weights
        total_samples = len(df)
        fall_weight = total_samples / (2 * fall_count)
        non_fall_weight = total_samples / (2 * non_fall_count)
        
        class_weights = {
            0: non_fall_weight,
            1: fall_weight
        }
        
        print(f"\nClass weights:")
        print(f"Fall class weight: {fall_weight:.2f}")
        print(f"Non-fall class weight: {non_fall_weight:.2f}")
        
        return df, class_weights
        
    except Exception as e:
        print(f"Error in balance_dataset_with_esc50: {e}")
        # Return original dataset with class weights if ESC-50 fails
        total_samples = len(df)
        fall_count = sum(df['label'] == 1)
        non_fall_count = sum(df['label'] == 0)
        
        fall_weight = total_samples / (2 * fall_count)
        non_fall_weight = total_samples / (2 * non_fall_count)
        
        class_weights = {
            0: non_fall_weight,
            1: fall_weight
        }
        
        print("\nUsing class weights only for balancing due to error")
        return df, class_weights

def analyze_esc50_impact(model, test_df, esc50_df):
    """
    Analyze the impact of ESC-50 samples on model performance.
    
    Args:
        model: Trained model
        test_df (pd.DataFrame): Test dataset
        esc50_df (pd.DataFrame): ESC-50 samples
    """
    print("\nAnalyzing ESC-50 impact on model performance:")
    
    # Get predictions for test set
    test_preds = model.predict(test_df.drop(['label', 'file_path', 'filename', 'fold', 'subject', 'environment', 'category', 'confidence'], axis=1))
    test_accuracy = accuracy_score(test_df['label'], test_preds)
    
    # Get predictions for ESC-50 samples
    esc50_preds = model.predict(esc50_df.drop(['label', 'file_path', 'filename', 'fold', 'subject', 'environment', 'category', 'confidence'], axis=1))
    esc50_accuracy = accuracy_score(esc50_df['label'], esc50_preds)
    
    # Analyze by category
    category_performance = {}
    for category in esc50_df['category'].unique():
        cat_df = esc50_df[esc50_df['category'] == category]
        cat_preds = model.predict(cat_df.drop(['label', 'file_path', 'filename', 'fold', 'subject', 'environment', 'category', 'confidence'], axis=1))
        cat_accuracy = accuracy_score(cat_df['label'], cat_preds)
        category_performance[category] = {
            'accuracy': cat_accuracy,
            'count': len(cat_df),
            'false_positives': sum((cat_preds == 1) & (cat_df['label'] == 0))
        }
    
    print(f"\nOverall Performance:")
    print(f"  Test set accuracy: {test_accuracy:.3f}")
    print(f"  ESC-50 accuracy: {esc50_accuracy:.3f}")
    
    print("\nCategory-wise Performance:")
    for category, metrics in category_performance.items():
        print(f"\n  {category}:")
        print(f"    Samples: {metrics['count']}")
        print(f"    Accuracy: {metrics['accuracy']:.3f}")
        print(f"    False Positives: {metrics['false_positives']}")
    
    # Identify problematic categories
    problematic_categories = [cat for cat, metrics in category_performance.items() 
                            if metrics['false_positives'] > metrics['count'] * 0.1]
    
    if problematic_categories:
        print("\nProblematic Categories (high false positive rate):")
        for category in problematic_categories:
            print(f"  - {category}")
        print("\nRecommendations:")
        print("  1. Review these categories for potential fall-like sounds")
        print("  2. Consider adding more specific features to distinguish falls")
        print("  3. Adjust model confidence threshold for these categories")

def save_results(ml_models_tuple, panns_model):
    """Save trained models and results."""
    try:
        print("\nSaving models and results...")
        
        # Create results directory if it doesn't exist
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Unpack ML models and calibrators
        if ml_models_tuple is not None:
            ml_models, calibrators = ml_models_tuple
            
            # Save ML models
            if ml_models is not None:
                ml_model_dir = os.path.join(RESULTS_DIR, "ml_models")
                os.makedirs(ml_model_dir, exist_ok=True)
                
                for model_name, model in ml_models.items():
                    model_path = os.path.join(ml_model_dir, f"{model_name}_model.joblib")
                    joblib.dump(model, model_path)
                    print(f"Saved {model_name} model to {model_path}")
            
            # Save calibrators
            if calibrators is not None:
                calibrator_dir = os.path.join(RESULTS_DIR, "calibrators")
                os.makedirs(calibrator_dir, exist_ok=True)
                
                for model_name, calibrator in calibrators.items():
                    calibrator_path = os.path.join(calibrator_dir, f"{model_name}_calibrator.joblib")
                    joblib.dump(calibrator, calibrator_path)
                    print(f"Saved {model_name} calibrator to {calibrator_path}")
        
        # Save PANNs model
        if panns_model is not None:
            panns_dir = os.path.join(RESULTS_DIR, "panns_model")
            os.makedirs(panns_dir, exist_ok=True)
            
            # Save model state
            model_path = os.path.join(panns_dir, "model_state.pt")
            torch.save(panns_model.state_dict(), model_path)
            print(f"Saved PANNs model state to {model_path}")
            
            # Save model config
            config_path = os.path.join(panns_dir, "model_config.json")
            config = {
                "model_type": "wavlm-base",
                "num_labels": 2,
                "device": str(device)
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Saved PANNs model config to {config_path}")
        
        print("\nAll models and results saved successfully!")
        return True
        
    except Exception as e:
        print(f"Error in save_results: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the fall detection system."""
    try:
        print("\n=== Starting Fall Detection System ===\n")
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        train_files, train_labels, val_files, val_labels = load_and_preprocess_data()
        
        # Calculate class weights
        class_counts = np.bincount(train_labels)
        total_samples = len(train_labels)
        class_weights = torch.FloatTensor(total_samples / (len(class_counts) * class_counts))
        class_weights = class_weights.to(device)
        print(f"Class weights: {class_weights}")
        
        # Train ML models
        print("\nTraining ML models...")
        ml_models_tuple = train_ml_models(train_files, train_labels, val_files, val_labels)
        if ml_models_tuple is None:
            raise ValueError("Failed to train ML models")
        
        ml_models, calibrators = ml_models_tuple
        
        # Initialize PANNs model and feature extractor
        print("\nInitializing PANNs model...")
        panns_model, panns_feature_extractor = load_panns_model()
        if panns_model is None:
            raise ValueError("Failed to load PANNs model")
            
        # Train PANNs model
        print("\nTraining PANNs model...")
        panns_model, train_losses, val_losses = train_panns_model(
            model=panns_model,
            feature_extractor=panns_feature_extractor,
            train_files=train_files,
            train_labels=train_labels,
            val_files=val_files,
            val_labels=val_labels,
            epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            output_dir="results/panns_finetuned",
            mixup_alpha=0.2,
            weight_decay=0.01,
            warmup_steps=100,
            gradient_accumulation_steps=1,
            class_weights=class_weights
        )
        
        # Save results
        print("\nSaving results...")
        if not save_results(ml_models_tuple, panns_model):
            raise ValueError("Failed to save results")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # 1. Model Architecture
        model_components = {
            "PANNs Model": ["WavLM Base", "Classification Head", "Feature Extractor"],
            "ML Models": ["Random Forest", "LightGBM", "Model Calibrators"],
            "Feature Extraction": ["Base Features", "Temporal Features", "Domain Features"],
            "Ensemble": ["Weighted Combination", "Uncertainty Estimation"]
        }
        viz.plot_model_architecture_diagram(model_components)
        
        # 2. Workflow Flowchart
        workflow_steps = {
            "Data Processing": [
                "Load Audio Files",
                "Extract Features",
                "Balance Dataset"
            ],
            "Model Training": [
                "Train ML Models",
                "Train PANNs Model",
                "Calibrate Models"
            ],
            "Ensemble": [
                "Combine Predictions",
                "Calculate Uncertainties",
                "Apply Post-processing"
            ],
            "Evaluation": [
                "Calculate Metrics",
                "Generate Visualizations",
                "Save Results"
            ]
        }
        viz.plot_workflow_flowchart(workflow_steps)
        
        # 3. Performance Metrics
        print("\nCalculating performance metrics...")
        # Get predictions for validation set
        val_features, _ = extract_features_for_ml(val_files, val_labels)
        if val_features is not None:
            # ML model predictions
            rf_probs = ml_models['rf'].predict_proba(val_features)[:, 1]
            lgb_probs = ml_models['lgb'].predict_proba(val_features)[:, 1]
            
            # PANNs predictions
            panns_probs = predict_with_panns(panns_model, panns_feature_extractor, val_files)
            if panns_probs is not None:
                panns_probs = panns_probs[:, 1]
                
                # Ensemble predictions
                ensemble_probs = (
                    0.4 * panns_probs +
                    0.3 * lgb_probs +
                    0.3 * rf_probs
                )
                
                # Plot performance metrics
                viz.plot_comprehensive_performance_metrics(
                    val_labels,
                    ensemble_probs,
                    class_names=['Non-Fall', 'Fall']
                )
                
                # Plot confusion matrix
                predictions = (ensemble_probs > 0.5).astype(int)
                viz.plot_research_grade_confusion_matrix(
                    val_labels,
                    predictions,
                    class_names=['Non-Fall', 'Fall']
                )
        
        # 4. Feature Importance
        print("\nAnalyzing feature importance...")
        feature_names = [f"feature_{i}" for i in range(val_features.shape[1])]
        importance_scores = [
            ml_models['rf'].feature_importances_,
            ml_models['lgb'].feature_importances_
        ]
        model_names = ['Random Forest', 'LightGBM']
        viz.plot_feature_importance_analysis(
            feature_names,
            importance_scores,
            model_names
        )
        
        # 5. Audio Analysis
        print("\nAnalyzing audio samples...")
        # Select a few representative samples
        sample_indices = np.random.choice(len(val_files), min(5, len(val_files)), replace=False)
        for idx in sample_indices:
            audio_path = val_files[idx]
            label = val_labels[idx]
            
            # Plot spectrogram analysis
            viz.plot_audio_spectrogram_analysis(
                audio_path,
                predictions=[ensemble_probs[idx]] if 'ensemble_probs' in locals() else None
            )
        
        # 6. Classification Reports
        print("\nGenerating classification reports...")
        if 'ensemble_probs' in locals():
            predictions = (ensemble_probs > 0.5).astype(int)
            print("\nClassification Report:")
            print(classification_report(val_labels, predictions))
            
            # Calculate additional metrics
            brier = brier_score_loss(val_labels, ensemble_probs)
            logloss = log_loss(val_labels, ensemble_probs)
            print(f"\nBrier Score: {brier:.4f}")
            print(f"Log Loss: {logloss:.4f}")
        
        print("\nTraining and visualization completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        print("\nError: Fall detection system failed")
        return False

def load_and_preprocess_data():
    """Load and preprocess the dataset for training."""
    try:
        print("\nLoading dataset...")
        train_df, val_df, test_df = load_dataset(DATA_DIR)
        if train_df is None or val_df is None or test_df is None:
            raise ValueError("Failed to load dataset")
            
        print(f"\nTrain/Validation samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        
        # Balance dataset and get class weights
        print("\nBalancing dataset...")
        balanced_df, class_weights = balance_dataset_with_esc50(train_df)
        if balanced_df is None:
            raise ValueError("Dataset balancing failed")
            
        print(f"\nFinal Training samples: {len(balanced_df)}")
        print(f"Final Validation samples: {len(val_df)}")
        
        # Print class distribution
        print("\nTraining set class distribution:")
        print(balanced_df['label'].value_counts())
        
        # Extract file paths and labels
        train_files = balanced_df['file_path'].values
        train_labels = balanced_df['label'].values
        val_files = val_df['file_path'].values
        val_labels = val_df['label'].values
        
        return train_files, train_labels, val_files, val_labels
        
    except Exception as e:
        print(f"Error in load_and_preprocess_data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

    # --- Q1 Visualization Script (Safe to run after main) ---
    import numpy as np
    import pandas as pd
    import os
    # Load results from the new CSV
    results_csv = os.path.join('results', 'combined_results.csv')
    df = pd.read_csv(results_csv)
    # Extract true_label, prediction, and fall_probability
    y_true = df['true_label'].values
    y_pred = df['prediction'].values
    y_prob = df['fall_probability'].values
    # Save as .npy for compatibility with visualization code
    np.save('results/y_true.npy', y_true)
    np.save('results/y_pred.npy', y_pred)
    np.save('results/y_prob.npy', y_prob)
    # Example audio files (update as needed)
    fall_audio = r'C:\Users\T2430477\Downloads\archive\audio_clips\10-181-05-079-01.wav'
    nonfall_audio = r'C:\Users\T2430477\Downloads\archive\audio_clips\10-189-00-143-02.wav'
    f1_scores = [0.89, 0.81, 0.78]  # Example F1 for [PANNs, LightGBM, RF]
    models = ['PANNs', 'LightGBM', 'Random Forest']

    from audio_fall_integrated import FallDetectionVisualizer
    viz = FallDetectionVisualizer()

    # 1. Model Architecture
    model_components = {
        "PANNs Model": ["WavLM Base", "Classification Head", "Feature Extractor"],
        "ML Models": ["Random Forest", "LightGBM", "Model Calibrators"],
        "Feature Extraction": ["Base Features", "Temporal Features", "Domain Features"],
        "Ensemble": ["Weighted Combination", "Uncertainty Estimation"]
    }
    viz.plot_model_architecture_diagram(model_components)

    # 2. Spectrograms
    viz.plot_audio_spectrogram_analysis(fall_audio, title='Fall Example')
    viz.plot_audio_spectrogram_analysis(nonfall_audio, title='Non-Fall Example')

    # 3. Confusion Matrix
    viz.plot_research_grade_confusion_matrix(y_true, y_pred, class_names=['Non-Fall', 'Fall'], title="Confusion Matrix")

    # 4/5. ROC & PR Curves
    viz.plot_comprehensive_performance_metrics(y_true, y_prob, class_names=['Non-Fall', 'Fall'], title="Performance Analysis")

    # 6. Metrics Table
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=['Non-Fall', 'Fall'], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv('results/performance_metrics_table.csv')
    print(df_report)

    # 7. (Optional) Attention Map: If you have attention_weights and audio_features, use advanced_visualizations.py
    # from advanced_visualizations import AudioVisualizationSuite
    # viz2 = AudioVisualizationSuite()
    # viz2.plot_attention_heatmap(attention_weights, audio_features)

    # 8. Model Comparison Bar Chart
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel('F1 Score')
    plt.title('Model Comparison')
    plt.ylim(0, 1)
    plt.savefig('results/model_comparison.png', dpi=300)
    plt.close()

