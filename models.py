import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import warnings
import os  # Add missing import
import librosa  # Add librosa import
import numpy as np  # Add numpy import for array operations
import matplotlib.pyplot as plt  # Add matplotlib for plotting

# Suppress specific warnings from transformers if needed
warnings.filterwarnings("ignore", message=".*Using a pipeline without specifying a model name and revision is deprecated.*")

def load_ast_model_for_finetuning(model_checkpoint="MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=2):
    """Loads a pre-trained Audio Spectrogram Transformer (AST) model from Hugging Face 
       and modifies its classifier head for binary classification (fine-tuning).

    Args:
        model_checkpoint (str): The Hugging Face model identifier for the pre-trained AST.
        num_labels (int): The number of output labels for the new classifier head (default: 2 for binary).

    Returns:
        tuple: (model, feature_extractor) 
               - model: The AST model instance ready for fine-tuning.
               - feature_extractor: The corresponding feature extractor for the model.
               Returns (None, None) if loading fails.
    """
    print(f"\nLoading AST model: {model_checkpoint}")
    try:
        # Load the feature extractor associated with the model
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
        print("Feature extractor loaded successfully.")

        # Load the pre-trained model
        model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, 
            num_labels=num_labels, 
            ignore_mismatched_sizes=True # Necessary to replace the classifier head
        )
        print("Pre-trained AST model loaded successfully.")

        # The classifier head is typically named 'classifier' or similar
        # For AST models, it's often model.classifier.dense for the final linear layer
        # We loaded it with num_labels=2 and ignore_mismatched_sizes=True, 
        # so the head should already be adapted for binary classification.
        # Verify the final layer shape
        if hasattr(model, 'classifier') and hasattr(model.classifier, 'out_features'):
             print(f"Classifier head output features: {model.classifier.out_features}")
             if model.classifier.out_features != num_labels:
                 print(f"Warning: Classifier head output features ({model.classifier.out_features}) do not match num_labels ({num_labels}). Re-check model architecture or loading parameters.")
        else:
             print("Warning: Could not automatically verify classifier head output size.")

        print("AST model ready for fine-tuning.")
        return model, feature_extractor

    except Exception as e:
        print(f"Error loading AST model or feature extractor from {model_checkpoint}: {e}")
        return None, None

# Example usage (for testing purposes):
if __name__ == '__main__':
    print("Testing AST model loading...")
    ast_model, ast_feature_extractor = load_ast_model_for_finetuning()
    
    if ast_model and ast_feature_extractor:
        print("\nAST Model Architecture (Top Level):")
        print(ast_model)
        print("\nAST Feature Extractor Config:")
        print(ast_feature_extractor)
        print("\nModel loading test successful.")
    else:
        print("\nModel loading test failed.")




# --- AST Fine-tuning Utilities ---
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm # For progress bars

# Assuming SR is defined globally or imported
try:
    from features import SR
except ImportError:
    SR = 22050 

class AudioDataset(Dataset):
    """PyTorch Dataset for loading audio files and preparing for AST."""
    def __init__(self, file_paths, labels, feature_extractor, max_length=int(16000 * 5), target_sr=16000, apply_mixup=False, mixup_alpha=0.2):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.target_sr = target_sr  # Set to 16000 Hz to match AST model requirements
        self.apply_mixup = apply_mixup
        self.mixup_alpha = mixup_alpha
        
        # Validate files exist
        self.valid_indices = []
        for i, file_path in enumerate(file_paths):
            if os.path.exists(file_path):
                self.valid_indices.append(i)
            else:
                print(f"Warning: File not found: {file_path}")
        
        if not self.valid_indices:
            raise ValueError("No valid audio files found!")
            
        print(f"Found {len(self.valid_indices)} valid files out of {len(file_paths)} total files")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        try:
            # Get the actual index from valid_indices
            actual_idx = self.valid_indices[idx]
            file_path = self.file_paths[actual_idx]
            label = self.labels[actual_idx]
            
            # Load and validate audio with detailed error checking
            try:
                # Load audio at original sample rate first
                audio, sr = librosa.load(file_path, sr=None, mono=True)
                print(f"\nProcessing {file_path}:")
                print(f"  Original audio shape: {audio.shape}, sample rate: {sr}")
                
                # Resample to target sample rate (16000 Hz)
                if sr != self.target_sr:
                    print(f"  Resampling from {sr} Hz to {self.target_sr} Hz")
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                    print(f"  Resampled audio shape: {audio.shape}")
                
                print(f"  Audio stats - min: {np.min(audio):.3f}, max: {np.max(audio):.3f}, mean: {np.mean(audio):.3f}, std: {np.std(audio):.3f}")
            except Exception as e:
                print(f"Error loading audio from {file_path}: {str(e)}")
                raise
                
            # Validate audio properties
            if audio is None or len(audio) == 0:
                raise ValueError(f"Failed to load audio from {file_path}")
                
            if len(audio) < self.target_sr * 0.5:  # Minimum 0.5 seconds
                raise ValueError(f"Audio too short: {len(audio)/self.target_sr:.2f}s")
                
            if np.max(np.abs(audio)) < 1e-6:
                raise ValueError(f"Audio is effectively silent: {file_path}")
                
            if not np.all(np.isfinite(audio)):
                raise ValueError(f"Audio contains NaN or Inf values: {file_path}")
                
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
                
            # Pad or truncate to max_length
            if len(audio) > self.max_length:
                print(f"  Truncating audio from {len(audio)/self.target_sr:.2f}s to {self.max_length/self.target_sr:.2f}s")
                audio = audio[:self.max_length]
            else:
                padding = np.zeros(self.max_length - len(audio))
                audio = np.concatenate([audio, padding])
                print(f"  Padded audio to {len(audio)/self.target_sr:.2f}s")
            
            # Apply mixup if enabled
            if self.apply_mixup and np.random.random() < 0.5:
                # Get another random sample
                mixup_idx = np.random.randint(0, len(self))
                mixup_actual_idx = self.valid_indices[mixup_idx]
                mixup_file = self.file_paths[mixup_actual_idx]
                mixup_label = self.labels[mixup_actual_idx]
                
                # Load mixup audio
                mixup_audio, mixup_sr = librosa.load(mixup_file, sr=None, mono=True)
                if mixup_audio is None or len(mixup_audio) == 0:
                    raise ValueError(f"Failed to load mixup audio from {mixup_file}")
                
                # Resample mixup audio if needed
                if mixup_sr != self.target_sr:
                    mixup_audio = librosa.resample(mixup_audio, orig_sr=mixup_sr, target_sr=self.target_sr)
                    
                # Pad/truncate mixup audio
                if len(mixup_audio) > self.max_length:
                    mixup_audio = mixup_audio[:self.max_length]
                else:
                    padding = np.zeros(self.max_length - len(mixup_audio))
                    mixup_audio = np.concatenate([mixup_audio, padding])
                
                # Apply mixup
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                audio = lam * audio + (1 - lam) * mixup_audio
                label = lam * label + (1 - lam) * mixup_label
                print(f"  Applied mixup with lambda={lam:.2f}")
            
            # Extract features using the feature extractor
            try:
                inputs = self.feature_extractor(
                    audio, 
                    sampling_rate=self.target_sr, 
                    return_tensors="pt"
                )
                
                # Ensure input_values has the correct shape for AST
                # AST expects [batch_size, sequence_length] for audio input
                if len(inputs["input_values"].shape) > 2:
                    inputs["input_values"] = inputs["input_values"].squeeze()
                
                print(f"  Successfully extracted features with shape: {inputs['input_values'].shape}")
            except Exception as e:
                print(f"Error extracting features from {file_path}: {str(e)}")
                raise
            
            # Convert label to one-hot encoding for binary classification
            if label == -1:  # Error case
                label_tensor = torch.tensor([0.0, 0.0], dtype=torch.float32)
            else:
                # Convert to one-hot encoding [normal, fall]
                label_tensor = torch.tensor([1.0 - label, label], dtype=torch.float32)
            
            inputs["labels"] = label_tensor
            
            return inputs
            
        except Exception as e:
            print(f"\nError processing {file_path}:")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            
            # Return a zero tensor with the correct shape
            inputs = self.feature_extractor(
                np.zeros(self.max_length), 
                sampling_rate=self.target_sr, 
                return_tensors="pt"
            )
            # Ensure correct shape for error case too
            if len(inputs["input_values"].shape) > 2:
                inputs["input_values"] = inputs["input_values"].squeeze()
            inputs["labels"] = torch.tensor([0.0, 0.0], dtype=torch.float32)  # Use [0,0] to mark error
            return inputs

def train_ast_model(
    model, 
    feature_extractor, 
    train_files, 
    train_labels, 
    val_files, 
    val_labels, 
    epochs=15,
    batch_size=8, 
    learning_rate=5e-5,
    output_dir="results/ast_finetuned",
    mixup_alpha=0.2,
    fall_class_weight=2.0,
    min_recall_threshold=0.95,
    patience=3,
    lr_scheduler_patience=2,
    lr_scheduler_factor=0.5,
    min_lr=1e-6
):
    """Fine-tunes an AST model for fall detection."""
    print("\n--- Fine-tuning AST Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize validation metrics
    val_recall = 0.0
    val_precision = 0.0
    val_f1 = 0.0
    val_auc = 0.0
    best_threshold = 0.5  # Default threshold
    
    # Create datasets
    train_dataset = AudioDataset(train_files, train_labels, feature_extractor, apply_mixup=True, mixup_alpha=mixup_alpha)
    val_dataset = AudioDataset(val_files, val_labels, feature_extractor, apply_mixup=False)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=lr_scheduler_patience,
        factor=lr_scheduler_factor,
        min_lr=min_lr
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, "best_model.pth")
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in progress_bar:
            # Move batch to device
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_values, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_steps += 1
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        all_val_preds = []
        all_val_labels = []
        
        progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                input_values = batch["input_values"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass
                outputs = model(input_values, labels=labels)
                loss = outputs.loss
                
                # Get predictions
                logits = outputs.logits
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                
                # Store predictions and labels (excluding error samples)
                valid_mask = labels.cpu().numpy().flatten() != -1
                all_val_preds.extend(probs[valid_mask])
                all_val_labels.extend(labels.cpu().numpy().flatten()[valid_mask])
                
                # Update metrics
                val_loss += loss.item()
                val_steps += 1
                progress_bar.set_postfix({"loss": loss.item()})
        
        avg_val_loss = val_loss / val_steps
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1} - Average Validation Loss: {avg_val_loss:.4f}")
        
        # Calculate validation metrics
        if len(all_val_preds) > 0 and len(all_val_labels) > 0:
            try:
                all_val_preds = np.array(all_val_preds)
                all_val_labels = np.array(all_val_labels)
                
                # Find optimal threshold
                precisions, recalls, thresholds = precision_recall_curve(all_val_labels, all_val_preds)
                valid_indices = np.where(recalls[:-1] >= min_recall_threshold)[0]
                
                if len(valid_indices) > 0:
                    best_threshold_idx = valid_indices[0]
                    best_threshold = thresholds[best_threshold_idx]
                    val_preds = (all_val_preds >= best_threshold).astype(int)
                    
                    # Calculate metrics
                    val_recall = recall_score(all_val_labels, val_preds)
                    val_precision = precision_score(all_val_labels, val_preds)
                    val_f1 = f1_score(all_val_labels, val_preds)
                    val_auc = roc_auc_score(all_val_labels, all_val_preds)
                    
                    print(f"\nValidation Metrics:")
                    print(f"  Recall: {val_recall:.4f}")
                    print(f"  Precision: {val_precision:.4f}")
                    print(f"  F1-Score: {val_f1:.4f}")
                    print(f"  AUC: {val_auc:.4f}")
                    print(f"  Optimal Threshold: {best_threshold:.4f}")
                else:
                    print(f"\nCould not achieve minimum recall threshold of {min_recall_threshold}")
            except Exception as e:
                print(f"Could not calculate validation metrics: {e}")
                print(f"Validation labels: {np.unique(all_val_labels)}")
                print(f"Validation predictions shape: {len(all_val_preds)}")
                print(f"Validation predictions range: [{min(all_val_preds):.4f}, {max(all_val_preds):.4f}]")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Model checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epochs")
            
            if patience_counter >= patience:
                print("\nEarly stopping triggered")
                break
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()
    
    print("\n--- AST Fine-tuning Complete ---")
    print("Best validation metrics achieved:")
    print(f"  Recall: {val_recall:.4f}")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  F1-Score: {val_f1:.4f}")
    print(f"  AUC: {val_auc:.4f}")
    print(f"  Optimal Threshold: {best_threshold:.4f}")
    print(f"Training curves saved to {os.path.join(output_dir, 'training_curves.png')}")
    
    # Load best model
    print("Loading best model from", output_dir)
    model.load_state_dict(torch.load(best_model_path))
    
    return model

def predict_with_ast(model, feature_extractor, file_paths, batch_size=8):
    """Generates predictions (probabilities) using a fine-tuned AST model."""
    print("\n--- Generating Predictions with Fine-tuned AST ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Validate input files and create mapping
    valid_files = []
    file_to_idx = {}  # Map from file path to original index
    for i, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        valid_files.append(file_path)
        file_to_idx[file_path] = i
    
    if not valid_files:
        print("Error: No valid files found for prediction")
        return np.array([]), []
    
    print(f"Found {len(valid_files)} valid files out of {len(file_paths)} total files")

    # Create a dataset without labels for prediction
    pred_dataset = AudioDataset(valid_files, [-1]*len(valid_files), feature_extractor, target_sr=feature_extractor.sampling_rate)
    pred_dataloader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_probs = []
    valid_indices = []
    error_files = []
    error_reasons = {}  # Store error reasons for each file
    progress_bar = tqdm(pred_dataloader, desc="Predicting")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Get the file paths for this batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(valid_files))
            batch_files = valid_files[start_idx:end_idx]
            
            try:
                # Move batch to device
                input_values = batch["input_values"].to(device)
                
                # Forward pass
                outputs = model(input_values)
                logits = outputs.logits
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                
                # Store predictions and valid indices
                all_probs.extend(probs)
                # Use the mapping to get original indices
                batch_indices = [file_to_idx[f] for f in batch_files]
                valid_indices.extend(batch_indices)
                
            except Exception as e:
                print(f"\nError processing batch {batch_idx}:")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Error message: {str(e)}")
                # Skip this batch but continue processing
                continue

    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    valid_indices = np.array(valid_indices)
    
    # Verify shapes match
    if len(all_probs) != len(valid_indices):
        print(f"Warning: Shape mismatch - predictions: {len(all_probs)}, indices: {len(valid_indices)}")
        # Take only the predictions that have corresponding indices
        min_len = min(len(all_probs), len(valid_indices))
        all_probs = all_probs[:min_len]
        valid_indices = valid_indices[:min_len]
    
    # Create a full-length array with NaN for failed predictions
    full_probs = np.full(len(file_paths), np.nan)
    full_probs[valid_indices] = all_probs
    
    return full_probs, valid_indices


