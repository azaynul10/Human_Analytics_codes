# audio_fall_integrated.py
# Main script integrating modular components for fall detection

import os
import numpy as np
import pandas as pd
import librosa
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
# --- Scikit-learn Imports ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_recall_curve, accuracy_score, precision_score, 
                           recall_score, f1_score, roc_auc_score, log_loss, brier_score_loss)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# --- LightGBM Import ---
import lightgbm as lgb

# --- PyTorch & Transformers Imports ---
import torch
# Note: Tensorflow/Keras imports removed as AST (PyTorch) replaces the CNN

# --- Custom Module Imports ---
# Assuming models.py, features.py, augmentations.py, error_analysis.py, optimization_utils.py 
# are in the same directory or Python path.
try:
    from models import load_ast_model_for_finetuning, train_ast_model, predict_with_ast
    from features import (extract_combined_features_for_ml, 
                          create_spectrogram_for_dl, SR, DURATION, N_MELS)
    from augmentations import (get_dl_augmentations, generate_augmented_audio_for_ml, 
                               load_rir_files)
    from error_analysis import analyze_and_save_errors
    from optimization_utils import (optimize_ensemble_weights, 
                                  plot_precision_recall_vs_threshold, 
                                  find_optimal_threshold)
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure models.py, features.py, augmentations.py, error_analysis.py, optimization_utils.py are accessible.")
    exit()

# --- Configuration --- 
# User Action Required: Set these paths and parameters

# Data Directories
BASE_DATA_DIR = r"C:\Users\T2430477\Downloads\archive" # Base directory containing the SAFE dataset structure
DATA_DIR = os.path.join(BASE_DATA_DIR, "audio_clips") # Directory with Fold 1-10 .wav files
ESC50_DIR = r"C:\Users\T2430477\Downloads\ESC-50-master\ESC-50-master\audio" # ESC-50 dataset directory
AUGMENTED_DATA_DIR = os.path.join(BASE_DATA_DIR, "augmented_audio_ml") # Dir to store augmented audio for ML
CACHE_DIR = os.path.join(BASE_DATA_DIR, "cache") # Directory for caching processed features
RIR_DATASET_DIR = r"C:\Users\T2430477\Downloads\air_database_release_1_4\AIR_1_4\AIR_wav_files" # Directory containing RIR .wav files

# Model & Output Directories
RESULTS_DIR = "results"
AST_OUTPUT_DIR = os.path.join(RESULTS_DIR, "ast_finetuned")
ML_MODEL_DIR = os.path.join(RESULTS_DIR, "ml_models")
ERROR_ANALYSIS_DIR = os.path.join(RESULTS_DIR, "error_analysis")

# AST Model Configuration
AST_MODEL_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"

# Training Parameters
TEST_FOLD = 10 # Which fold to use as the final test set
AST_EPOCHS = 30  # Increased from 25 to 30 for better convergence
AST_BATCH_SIZE = 16  # Reduced from 32 for better gradient updates
AST_LEARNING_RATE = 2e-5  # Reduced from 3e-5 for more stable training
AST_WEIGHT_DECAY = 1e-4  # Added weight decay for regularization
ML_BATCH_SIZE = 32 # Batch size for ML models (if applicable, often not needed)
SEED = 42

# Enhanced Augmentation Parameters
NUM_ML_AUGMENTATIONS = {
    'fall': 10,  # Increased from 8 to 10
    'non_fall': 5
}
APPLY_AUGMENTATION_ML = True # Generate augmented data for ML models
APPLY_AUGMENTATION_DL = True # Apply on-the-fly augmentation for AST
AUGMENTATION_PROBABILITY = 0.7  # Increased from 0.6 to 0.7

# Augmentation Parameters for Different Sound Types
AUGMENTATION_CONFIG = {
    'fall': {
        'time_stretch': (0.7, 1.3),  # More aggressive time stretching
        'pitch_shift': (-3, 3),      # More aggressive pitch shifting
        'noise_level': (0.01, 0.08), # Higher noise levels
        'rir_probability': 0.8,      # Higher chance of room simulation
        'gain_range': (-4, 4)        # More gain variation
    },
    'non_fall': {
        'time_stretch': (0.9, 1.1),  # More conservative time stretching
        'pitch_shift': (-1, 1),      # More conservative pitch shifting
        'noise_level': (0.005, 0.02),# Less noise for normal sounds
        'rir_probability': 0.5,      # Lower chance of room simulation
        'gain_range': (-2, 2)        # More conservative gain adjustment
    }
}

# Class Weighting Configuration
FALL_CLASS_WEIGHT = 10.0  # Increased for higher recall, prioritize not missing falls
NORMAL_CLASS_WEIGHT = 1.0  # Weight for normal class (class 0)

# Ensemble Configuration
ENSEMBLE_WEIGHTS = {
    'ast': 0.65,  # Increased from 0.6 to 0.65
    'lgb': 0.20,  # Reduced from 0.25 to 0.20
    'rf': 0.15    # Reduced from 0.15 to 0.15
}
OPTIMIZE_WEIGHTS = True # Whether to run weight optimization on validation set
OPTIMIZE_THRESHOLD_TARGET = "recall" # Target metric for threshold optimization
MIN_RECALL_CONSTRAINT = 1.0 # Set to 1.0 to never miss a fall
MAX_FALSE_POSITIVE_RATE = 0.10 # Reduced from 0.15 to 0.10 for fewer false alarms

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
EARLY_STOPPING_PATIENCE = 8  # Increased from 5 to 8
EARLY_STOPPING_DELTA = 0.001  # Added minimum improvement threshold

# Learning Rate Schedule
LR_SCHEDULER_FACTOR = 0.5  # Added learning rate reduction factor
LR_SCHEDULER_PATIENCE = 3  # Added patience for learning rate reduction
LR_SCHEDULER_MIN_LR = 1e-6  # Added minimum learning rate

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

# --- Utility Functions ---

def load_audio_file(file_path, target_sr=SR):
    """Loads an audio file using librosa, with robust validation.
    
    Args:
        file_path (str): Path to the audio file.
        target_sr (int): Target sample rate.
        
    Returns:
        tuple: (audio, sr) or (None, None) if loading fails.
    """
    try:
        print(f"\nLoading audio file: {file_path}")
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True, res_type='kaiser_fast')
        print(f"Loaded audio shape: {audio.shape}, sample rate: {sr}")
        
        # Check for empty or invalid audio
        if audio is None or len(audio) == 0:
            print(f"Error: Loaded audio is empty: {file_path}")
            return None, None
            
        if not np.all(np.isfinite(audio)):
            print(f"Error: Audio contains NaN or Inf values: {file_path}")
            return None, None
        
        # Enforce minimum length (0.5 seconds for robust feature extraction)
        min_length = target_sr * 0.5
        if len(audio) < min_length:
            print(f"Error: Audio too short ({len(audio)/sr:.2f}s < {min_length/sr:.2f}s): {file_path}")
            return None, None
        
        # Check for silence
        if np.max(np.abs(audio)) < 1e-6:
            print(f"Error: Audio is effectively silent (max amplitude: {np.max(np.abs(audio)):.2e}): {file_path}")
            return None, None
        
        # Check for DC offset
        dc_offset = np.mean(audio)
        if abs(dc_offset) > 0.1:  # Threshold for DC offset
            print(f"Warning: Audio has significant DC offset ({dc_offset:.3f}): {file_path}")
            audio = audio - dc_offset
        
        # Check for clipping
        if np.max(np.abs(audio)) > 0.99:
            print(f"Warning: Audio may be clipped (max amplitude: {np.max(np.abs(audio)):.3f}): {file_path}")
        
        # Convert to float32 and normalize
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Print audio statistics
        print(f"Audio statistics:")
        print(f"  Duration: {len(audio)/sr:.2f}s")
        print(f"  Min: {np.min(audio):.3f}")
        print(f"  Max: {np.max(audio):.3f}")
        print(f"  Mean: {np.mean(audio):.3f}")
        print(f"  Std: {np.std(audio):.3f}")
        print(f"  RMS: {np.sqrt(np.mean(np.square(audio))):.3f}")
        
        return audio, sr
        
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def parse_filename(filename):
    """Parse the structured filename format: AA-BBB-CC-DDD-FF.wav"""
    try:
        parts = os.path.splitext(filename)[0].split('-')
        if len(parts) != 5:
            return None
        fold = int(parts[0])
        subject = int(parts[1]) # Assuming BBB is subject ID
        env = int(parts[2])     # Assuming CC is environment ID
        seq = int(parts[3])
        label = int(parts[4]) # 1=normal, 2=fall
        return fold, subject, env, seq, label
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return None

def prepare_and_cache_features(file_path, use_cache=True):
    """Loads audio, extracts ML features, and caches the result."""
    cache_filename = os.path.basename(file_path) + ".pkl"
    cache_path = os.path.join(CACHE_DIR, cache_filename)

    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            if len(cached_data) != 1747:
                print(f"Cache error: Invalid feature length in {cache_filename} ({len(cached_data)}). Deleting cache file.")
                os.remove(cache_path)
                return None
            return cached_data
        except Exception as e:
            print(f"Cache read error for {cache_filename}: {e}. Deleting invalid cache file.")
            try:
                os.remove(cache_path)
            except Exception as del_e:
                print(f"Warning: Could not delete invalid cache file {cache_filename}: {del_e}")
            return None

    audio, sr = load_audio_file(file_path, target_sr=SR)
    if audio is None:
        print(f"Skipping {file_path}: Audio loading failed")
        return None

    ml_features = extract_combined_features_for_ml(audio, sr)
    if ml_features is None or len(ml_features) != 1747:
        print(f"Skipping {file_path}: Feature extraction failed or invalid length")
        return None

    if use_cache:
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(ml_features, f)
        except Exception as e:
            print(f"Cache write error for {cache_filename}: {e}")
    
    return ml_features

def load_dataset(data_dir, test_fold=TEST_FOLD):
    """Loads the dataset, parses filenames, and prepares data splits."""
    print(f"\nLoading dataset from: {data_dir}")
    all_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".wav")]
    print(f"Found {len(all_files)} .wav files in main dataset.")

    data = []
    for filename in tqdm(all_files, desc="Parsing main dataset filenames"): 
        file_path = os.path.join(data_dir, filename)
        metadata = parse_filename(filename)
        if metadata:
            fold, subject, env, _, label = metadata
            if label in [1, 2]: # Valid labels
                mapped_label = label - 1 # 0 for normal, 1 for fall
                data.append({
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

    # Add relevant ESC-50 files as normal examples
    print(f"\nLoading ESC-50 dataset from: {ESC50_DIR}")
    esc50_files = [f for f in os.listdir(ESC50_DIR) if f.lower().endswith(".wav")]
    print(f"Found {len(esc50_files)} total .wav files in ESC-50 dataset.")

    # Add filtered ESC-50 files as normal examples
    for filename in tqdm(esc50_files, desc="Adding relevant ESC-50 files"):
        # Extract category from filename (ESC-50 format: category-filename.wav)
        category = filename.split('-')[0].lower()
        
        # Only include if category is in our relevant list
        if category in RELEVANT_ESC50_CATEGORIES:
            file_path = os.path.join(ESC50_DIR, filename)
            data.append({
                "file_path": file_path,
                "filename": filename,
                "fold": 0,  # Special fold for ESC-50 data
                "subject": -1,  # No subject ID for ESC-50
                "environment": -1,  # No environment ID for ESC-50
                "label": 0,  # All ESC-50 files are labeled as normal
                "category": category,
                "confidence": RELEVANT_ESC50_CATEGORIES[category]  # Use category-specific confidence
            })

    # --- Add Le2i Dataset Integration: Separate Normal and Fall Folders ---

    LE2I_NORMAL_DIR = r"C:\Users\T2430477\Downloads\raw\raw\audio\normal"
    LE2I_FALL_DIR = r"C:\Users\T2430477\Downloads\raw\raw\audio\abnormal"

    # Add normal (non-fall) files
    if os.path.exists(LE2I_NORMAL_DIR):
        normal_files = [f for f in os.listdir(LE2I_NORMAL_DIR) if f.lower().endswith('.wav')]
        print(f"Found {len(normal_files)} Le2i normal audio files.")
        for filename in normal_files:
            file_path = os.path.join(LE2I_NORMAL_DIR, filename)
            data.append({
                "file_path": file_path,
                "filename": filename,
                "fold": 0,  # Or another special fold if you want to track source
                "subject": -1,
                "environment": -1,
                "label": 0,  # Non-fall
                "category": "le2i_normal",
                "confidence": 1.0
            })
    else:
        print(f"Le2i normal audio directory not found: {LE2I_NORMAL_DIR}")

    # Add fall (abnormal) files
    if os.path.exists(LE2I_FALL_DIR):
        fall_files = [f for f in os.listdir(LE2I_FALL_DIR) if f.lower().endswith('.wav')]
        print(f"Found {len(fall_files)} Le2i fall audio files.")
        for filename in fall_files:
            file_path = os.path.join(LE2I_FALL_DIR, filename)
            data.append({
                "file_path": file_path,
                "filename": filename,
                "fold": 0,  # Or another special fold if you want to track source
                "subject": -1,
                "environment": -1,
                "label": 1,  # Fall
                "category": "le2i_fall",
                "confidence": 1.0
            })
    else:
        print(f"Le2i fall audio directory not found: {LE2I_FALL_DIR}")

    df = pd.DataFrame(data)
    print(f"\nTotal dataset size: {len(df)} samples")
    print(f"Main dataset samples: {len(all_files)}")
    print(f"Relevant ESC-50 samples included: {len(df[df['fold'] == 0])}")
    print(f"\nClass distribution:")
    print(df["label"].value_counts())
    
    # Only print ESC-50 category distribution if there are ESC-50 samples
    if len(df[df['fold'] == 0]) > 0 and 'category' in df.columns:
        print("\nESC-50 category distribution:")
        print(df[df['fold'] == 0]['category'].value_counts())

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

    # --- Diagnostics: Check Le2i and Augmented Files in Splits ---
    def count_category(df, keyword):
        if 'category' in df.columns:
            return df['category'].str.contains(keyword, case=False, na=False).sum()
        return 0
    def count_augmented(df):
        return df['filename'].str.contains('aug', case=False, na=False).sum()

    print("\n[Diagnostics] Le2i files in splits:")
    print(f"  Train: {count_category(train_df, 'le2i')}")
    print(f"  Val:   {count_category(val_df, 'le2i')}")
    print(f"  Test:  {count_category(test_df, 'le2i')}")
    print("[Diagnostics] Augmented files in splits:")
    print(f"  Train: {count_augmented(train_df)}")
    print(f"  Val:   {count_augmented(val_df)}")
    print(f"  Test:  {count_augmented(test_df)}")

    return train_df, val_df, test_df

# --- ML Model Training & Prediction ---
def train_ml_models(train_df, val_df):
    print("\n--- Training ML Models (LGBM, RF) with Enhanced Augmentation ---")
    
    # Prepare training data
    X_train_ml = []
    y_train_ml = []
    train_files_to_process = list(train_df["file_path"])
    rir_files = load_rir_files(RIR_DATASET_DIR) if APPLY_AUGMENTATION_ML else []

    if APPLY_AUGMENTATION_ML:
        print(f"Generating augmented versions with targeted strategies...")
        augmented_train_files = []
        augmented_labels = []
        
        # Process each file with appropriate augmentation strategy
        for i, file_path in enumerate(tqdm(train_files_to_process, desc="Generating ML Augmentations")):
            label = train_df["label"].iloc[i]
            sound_type = 'fall' if label == 1 else 'non_fall'
            num_augs = NUM_ML_AUGMENTATIONS[sound_type]
            aug_config = AUGMENTATION_CONFIG[sound_type]
            
            # Generate augmented versions with specific config
            augmented_paths = generate_augmented_audio_for_ml(
                file_path, 
                AUGMENTED_DATA_DIR, 
                rir_files, 
                num_augs,
                time_stretch_range=aug_config['time_stretch'],
                pitch_shift_range=aug_config['pitch_shift'],
                noise_level_range=aug_config['noise_level'],
                rir_probability=aug_config['rir_probability'],
                gain_range=aug_config['gain_range']
            )
            
            augmented_train_files.extend(augmented_paths)
            augmented_labels.extend([label] * len(augmented_paths))
        
        # Add original files and their labels
        train_files_to_process.extend(augmented_train_files)
        y_train_ml.extend(list(train_df["label"]) + augmented_labels)
        
        print(f"\nAugmentation Statistics:")
        print(f"Original files: {len(train_df)}")
        print(f"Augmented files: {len(augmented_train_files)}")
        print(f"Total files: {len(train_files_to_process)}")
        print(f"Class distribution after augmentation:")
        print(pd.Series(y_train_ml).value_counts())
    else:
        y_train_ml = list(train_df["label"])

    with ThreadPoolExecutor(max_workers=joblib.parallel.DEFAULT_N_JOBS) as executor:
        futures = [executor.submit(prepare_and_cache_features, fp) for fp in train_files_to_process]
        for i, future in tqdm(enumerate(futures), total=len(futures), desc="Extracting ML Train Features"):
            features = future.result()
            if features is not None and len(features) == 1747:
                X_train_ml.append(features.flatten())
            else:
                print(f"Warning: Skipping training file {train_files_to_process[i]} due to invalid features")
                y_train_ml.pop(len(X_train_ml))  # Remove corresponding label

    if not X_train_ml:
        print("Error: No valid training features extracted. Exiting.")
        exit()
    X_train_ml = np.array(X_train_ml)
    y_train_ml = np.array(y_train_ml[:len(X_train_ml)])  # Ensure alignment

    # Prepare validation data
    X_val_ml = []
    y_val_ml = list(val_df["label"])
    with ThreadPoolExecutor(max_workers=joblib.parallel.DEFAULT_N_JOBS) as executor:
        futures = [executor.submit(prepare_and_cache_features, fp) for fp in val_df["file_path"]]
        for i, future in tqdm(enumerate(futures), total=len(futures), desc="Extracting ML Val Features"):
            features = future.result()
            if features is not None and len(features) == 1747:
                X_val_ml.append(features.flatten())
            else:
                print(f"Warning: Skipping validation file {val_df['file_path'].iloc[i]} due to invalid features")
                y_val_ml.pop(len(X_val_ml))

    if not X_val_ml:
        print("Error: No valid validation features extracted. Exiting.")
        exit()
    X_val_ml = np.array(X_val_ml)
    y_val_ml = np.array(y_val_ml[:len(X_val_ml)])

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_ml)
    X_val_scaled = scaler.transform(X_val_ml)
    scaler_path = os.path.join(ML_MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # Calculate class weights based on class distribution
    n_samples = len(y_train_ml)
    n_falls = np.sum(y_train_ml == 1)
    n_normal = n_samples - n_falls
    print(f"\nClass distribution in training set:")
    print(f"Normal samples: {n_normal}")
    print(f"Fall samples: {n_falls}")
    print(f"Ratio (normal/fall): {n_normal/n_falls:.2f}")

    # Optuna optimization for LightGBM
    print("\nOptimizing LightGBM parameters with Optuna...")
    import optuna
    from optuna.integration import LightGBMPruningCallback

    def objective_lgb(trial):
        param = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': FALL_CLASS_WEIGHT,
            'random_state': SEED
        }
        
        model = lgb.LGBMClassifier(**param)
        model.fit(
            X_train_scaled, 
            y_train_ml,
            eval_set=[(X_val_scaled, y_val_ml)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        # Get predictions and calculate metrics
        y_pred = model.predict(X_val_scaled)
        recall = recall_score(y_val_ml, y_pred)
        precision = precision_score(y_val_ml, y_pred)
        f1 = f1_score(y_val_ml, y_pred)
        
        # Optimize for F1 score while maintaining high recall
        if recall < MIN_RECALL_CONSTRAINT:
            return 0.0  # Penalize if recall is too low
        
        return f1

    # Create and run the study
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=True)
    
    # Get best parameters and train final model
    best_params_lgb = study_lgb.best_params
    best_params_lgb.update({
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'scale_pos_weight': FALL_CLASS_WEIGHT,
        'random_state': SEED
    })
    
    print("\nBest LightGBM parameters:")
    print(best_params_lgb)
    
    lgb_model = lgb.LGBMClassifier(**best_params_lgb)
    lgb_model.fit(
        X_train_scaled, 
        y_train_ml,
        eval_set=[(X_val_scaled, y_val_ml)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    lgb_path = os.path.join(ML_MODEL_DIR, "lgb_model.pkl")
    joblib.dump(lgb_model, lgb_path)

    # Optuna optimization for Random Forest
    print("\nOptimizing Random Forest parameters with Optuna...")
    
    def objective_rf(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 10, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'bootstrap': True,
            'class_weight': {0: NORMAL_CLASS_WEIGHT, 1: FALL_CLASS_WEIGHT},
            'random_state': SEED,
            'n_jobs': joblib.parallel.DEFAULT_N_JOBS
        }
        
        model = RandomForestClassifier(**param)
        model.fit(X_train_scaled, y_train_ml)
        
        # Get predictions and calculate metrics
        y_pred = model.predict(X_val_scaled)
        recall = recall_score(y_val_ml, y_pred)
        precision = precision_score(y_val_ml, y_pred)
        f1 = f1_score(y_val_ml, y_pred)
        
        # Optimize for F1 score while maintaining high recall
        if recall < MIN_RECALL_CONSTRAINT:
            return 0.0  # Penalize if recall is too low
        
        return f1

    # Create and run the study
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(objective_rf, n_trials=50, show_progress_bar=True)
    
    # Get best parameters and train final model
    best_params_rf = study_rf.best_params
    best_params_rf.update({
        'bootstrap': True,
        'class_weight': {0: NORMAL_CLASS_WEIGHT, 1: FALL_CLASS_WEIGHT},
        'random_state': SEED,
        'n_jobs': joblib.parallel.DEFAULT_N_JOBS
    })
    
    print("\nBest Random Forest parameters:")
    print(best_params_rf)
    
    rf_model = RandomForestClassifier(**best_params_rf)
    rf_model.fit(X_train_scaled, y_train_ml)
    rf_path = os.path.join(ML_MODEL_DIR, "rf_model.pkl")
    joblib.dump(rf_model, rf_path)

    # Save optimization results
    optuna_results = {
        'lightgbm': {
            'best_params': best_params_lgb,
            'best_value': study_lgb.best_value,
            'best_trial': study_lgb.best_trial.number
        },
        'random_forest': {
            'best_params': best_params_rf,
            'best_value': study_rf.best_value,
            'best_trial': study_rf.best_trial.number
        }
    }
    
    results_path = os.path.join(ML_MODEL_DIR, "optuna_results.json")
    with open(results_path, 'w') as f:
        json.dump(optuna_results, f, indent=4)

    print("--- ML Model Training Complete ---")
    return lgb_model, rf_model, scaler

def predict_ml_models(lgb_model, rf_model, scaler, file_paths):
    """Generates predictions using trained ML models."""
    print(f"\n--- Predicting with ML Models for {len(file_paths)} files ---")
    X_ml = []
    valid_indices = [] # Keep track of files processed successfully
    
    with ThreadPoolExecutor(max_workers=joblib.parallel.DEFAULT_N_JOBS) as executor:
        futures = {executor.submit(prepare_and_cache_features, fp): i for i, fp in enumerate(file_paths)}
        results = {} # Store results mapped by original index
        for future in tqdm(futures.keys(), total=len(futures), desc="Extracting ML Features for Prediction"):
            original_index = futures[future]
            features = future.result()
            if features is not None:
                results[original_index] = features
            else:
                print(f"Warning: Feature extraction failed for prediction file index {original_index}. Skipping.")

    # Ensure features are ordered correctly
    ordered_features = [results[i] for i in sorted(results.keys())]
    valid_indices = sorted(results.keys())
    
    if not ordered_features:
        print("Error: No features extracted for ML prediction.")
        # Return empty arrays with expected structure
        return np.full(len(file_paths), np.nan), np.full(len(file_paths), np.nan), []

    X_ml = np.array(ordered_features)
    X_scaled = scaler.transform(X_ml)

    # Get predictions
    lgb_probs = lgb_model.predict_proba(X_scaled)[:, 1]
    rf_probs = rf_model.predict_proba(X_scaled)[:, 1]

    # Create full-length arrays with NaN for failed predictions
    full_lgb_probs = np.full(len(file_paths), np.nan)
    full_rf_probs = np.full(len(file_paths), np.nan)
    
    # Fill in the valid predictions
    full_lgb_probs[valid_indices] = lgb_probs
    full_rf_probs[valid_indices] = rf_probs

    return full_lgb_probs, full_rf_probs, valid_indices

# --- AST Fine-tuning & Prediction ---
# Uses functions from models.py

# --- Ensemble Prediction ---
def predict_ensemble(ast_probs, lgb_probs, rf_probs, weights):
    """Combines predictions using weighted averaging."""
    if not isinstance(weights, dict):
         raise ValueError('Weights must be a dictionary, e.g., {"AST": 0.4, "LGB": 0.3, "RF": 0.3}')
    # Ensure all probability arrays have the same length
    min_len = min(len(ast_probs), len(lgb_probs), len(rf_probs))
    if len(ast_probs) != min_len or len(lgb_probs) != min_len or len(rf_probs) != min_len:
        print("Warning: Probability array lengths differ. Truncating to minimum length.")
        ast_probs = ast_probs[:min_len]
        lgb_probs = lgb_probs[:min_len]
        rf_probs = rf_probs[:min_len]
    ensemble_prob = (
        weights.get("AST", 0) * ast_probs + 
        weights.get("LGB", 0) * lgb_probs + 
        weights.get("RF", 0) * rf_probs
    )
    # Ensure weights sum to approx 1, otherwise normalize?
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
        print(f"Warning: Ensemble weights do not sum to 1 (sum={total_weight:.2f}). Normalizing probabilities might be affected.")
        # Optionally normalize: ensemble_prob /= total_weight
    return ensemble_prob

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

def balance_dataset_with_esc50(df, target_ratio=0.7):
    """
    Balance the dataset using only real data from the main dataset.
    Removes all synthetic data and ensures proper class balance.
    
    Args:
        df (pd.DataFrame): The dataset DataFrame
        target_ratio (float): Target ratio of normal to fall samples (default: 0.7)
        
    Returns:
        pd.DataFrame: Balanced dataset containing only real data
    """
    # Remove all synthetic data (fold == 0)
    real_data = df[df['fold'] != 0].copy()
    
    # Count fall samples
    n_falls = len(real_data[real_data['label'] == 1])
    
    # Calculate target number of normal samples
    target_normals = int(n_falls / (1 - target_ratio) - n_falls)
    
    # Get current normal samples
    current_normals = len(real_data[real_data['label'] == 0])
    
    # Calculate how many more normal samples we need
    needed_normals = target_normals - current_normals
    
    if needed_normals <= 0:
        print("No additional normal samples needed")
        return real_data
    
    # Sort normal samples by confidence if available, otherwise random
    normal_samples = real_data[real_data['label'] == 0]
    if 'confidence' in normal_samples.columns:
        normal_samples = normal_samples.sort_values('confidence', ascending=False)
    else:
        normal_samples = normal_samples.sample(frac=1, random_state=42)  # Random shuffle
    
    # Select the required number of normal samples
    selected_normals = normal_samples.head(target_normals)
    
    # Combine fall samples with selected normal samples
    fall_samples = real_data[real_data['label'] == 1]
    balanced_df = pd.concat([fall_samples, selected_normals])
    
    # Print detailed balancing statistics
    print(f"\nDataset balancing statistics:")
    print(f"  Original fall samples: {n_falls}")
    print(f"  Original normal samples: {current_normals}")
    print(f"  Target normal samples: {target_normals}")
    print(f"  Selected normal samples: {len(selected_normals)}")
    print(f"  Final normal samples: {len(balanced_df[balanced_df['label'] == 0])}")
    
    # Verify final class balance
    final_ratio = len(balanced_df[balanced_df['label'] == 0]) / len(balanced_df[balanced_df['label'] == 1])
    print(f"\nFinal class ratio (normal/fall): {final_ratio:.2f}")
    print(f"Target ratio: {target_ratio:.2f}")
    
    return balanced_df

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

# --- Main Execution Logic --- 
def main():
    print("===== Starting Integrated Fall Detection Pipeline ====")
    start_time = datetime.now()

    # Clear cache directory to ensure fresh feature extraction
    if not clear_cache_directory():
        print("Warning: Could not clear cache directory. Proceeding with existing cache.")

    # 1. Load Data Splits
    train_df, val_df, test_df = load_dataset(DATA_DIR, test_fold=TEST_FOLD)
    
    if train_df.empty or val_df.empty or test_df.empty:
        print("Error: Data loading resulted in empty dataframes. Check DATA_DIR and parsing logic.")
        return

    # After loading the dataset, balance it
    df = balance_dataset_with_esc50(pd.concat([train_df, val_df, test_df]))

    # 2. Train ML Models (LGBM, RF) with Augmentation
    lgb_model, rf_model, scaler = train_ml_models(train_df, val_df)

    # 3. Fine-tune AST Model
    print("\n--- Loading and Fine-tuning AST Model ---")
    ast_model_base, feature_extractor = load_ast_model_for_finetuning(AST_MODEL_CHECKPOINT)
    if ast_model_base is None or feature_extractor is None:
        print("Error: Failed to load base AST model. Exiting.")
        return
        
    # Prepare file lists and labels for AST training
    train_files_ast = list(train_df["file_path"])
    train_labels_ast = list(train_df["label"])
    val_files_ast = list(val_df["file_path"])
    val_labels_ast = list(val_df["label"])
    
    # Add data validation checks
    print("\nValidating training data:")
    print(f"Number of training files: {len(train_files_ast)}")
    print(f"Number of validation files: {len(val_files_ast)}")
    print(f"Training label distribution: {pd.Series(train_labels_ast).value_counts().to_dict()}")
    print(f"Validation label distribution: {pd.Series(val_labels_ast).value_counts().to_dict()}")
    
    # Verify file existence
    train_files_exist = [os.path.exists(f) for f in train_files_ast]
    val_files_exist = [os.path.exists(f) for f in val_files_ast]
    print(f"Training files exist: {sum(train_files_exist)}/{len(train_files_exist)}")
    print(f"Validation files exist: {sum(val_files_exist)}/{len(val_files_exist)}")
    
    if not all(train_files_exist) or not all(val_files_exist):
        print("Error: Some audio files are missing. Please check the file paths.")
        return
    
    # Verify audio loading
    print("\nTesting audio loading for a few samples:")
    test_files = train_files_ast[:3] + val_files_ast[:3]
    for f in test_files:
        try:
            audio, sr = librosa.load(f, sr=feature_extractor.sampling_rate)
            print(f"Successfully loaded {f}: shape={audio.shape}, sr={sr}")
        except Exception as e:
            print(f"Error loading {f}: {e}")
            return
    
    ast_model_tuned = train_ast_model(
        model=ast_model_base,
        feature_extractor=feature_extractor,
        train_files=train_files_ast,
        train_labels=train_labels_ast,
        val_files=val_files_ast,
        val_labels=val_labels_ast,
        epochs=AST_EPOCHS,
        learning_rate=AST_LEARNING_RATE,
        lr_scheduler_patience=2,
        lr_scheduler_factor=0.5,
        min_lr=1e-6,
        patience=3,
        fall_class_weight=FALL_CLASS_WEIGHT,
        min_recall_threshold=MIN_RECALL_CONSTRAINT
    )

    # 4. Evaluate on Validation Set & Optimize Ensemble
    print("\n--- Evaluating on Validation Set and Optimizing Ensemble ---")
    val_file_paths = list(val_df["file_path"])
    y_val_true = np.array(val_df["label"])

    # Get predictions from each model on the validation set
    ast_val_probs, ast_valid_indices = predict_with_ast(ast_model_tuned, feature_extractor, val_file_paths, batch_size=AST_BATCH_SIZE)
    lgb_val_probs, rf_val_probs, ml_valid_indices = predict_ml_models(lgb_model, rf_model, scaler, val_file_paths)

    # Find common valid indices between AST and ML predictions
    common_valid_indices = np.intersect1d(ast_valid_indices, ml_valid_indices)
    if len(common_valid_indices) == 0:
        print("Error: No common valid predictions between AST and ML models. Cannot proceed with ensemble optimization.")
        return

    # Align validation predictions using common valid indices
    ast_val_probs_aligned = ast_val_probs[common_valid_indices]
    lgb_val_probs_aligned = lgb_val_probs[common_valid_indices]
    rf_val_probs_aligned = rf_val_probs[common_valid_indices]
    y_val_true_aligned = y_val_true[common_valid_indices]
    val_file_paths_aligned = [val_file_paths[i] for i in common_valid_indices]

    print(f"\nAligned validation set size: {len(common_valid_indices)}")
    print(f"AST valid predictions: {len(ast_valid_indices)}")
    print(f"ML valid predictions: {len(ml_valid_indices)}")
    print(f"Common valid predictions: {len(common_valid_indices)}")

    # Calculate validation metrics using aligned predictions
    try:
        # Find optimal threshold
        precisions, recalls, thresholds = precision_recall_curve(y_val_true_aligned, ast_val_probs_aligned)
        valid_indices = np.where(recalls[:-1] >= MIN_RECALL_CONSTRAINT)[0]
        
        if len(valid_indices) > 0:
            best_threshold_idx = valid_indices[0]
            best_threshold = thresholds[best_threshold_idx]
            val_preds = (ast_val_probs_aligned >= best_threshold).astype(int)
            
            # Calculate metrics
            val_recall = recall_score(y_val_true_aligned, val_preds)
            val_precision = precision_score(y_val_true_aligned, val_preds)
            val_f1 = f1_score(y_val_true_aligned, val_preds)
            val_auc = roc_auc_score(y_val_true_aligned, ast_val_probs_aligned)
            
            print(f"\nValidation Metrics:")
            print(f"  Recall: {val_recall:.4f}")
            print(f"  Precision: {val_precision:.4f}")
            print(f"  F1-Score: {val_f1:.4f}")
            print(f"  AUC: {val_auc:.4f}")
            print(f"  Optimal Threshold: {best_threshold:.4f}")
        else:
            print(f"\nCould not achieve minimum recall threshold of {MIN_RECALL_CONSTRAINT}")
    except Exception as e:
        print(f"Could not calculate validation metrics: {e}")
        print(f"Validation labels: {np.unique(y_val_true_aligned)}")
        print(f"Validation predictions shape: {len(ast_val_probs_aligned)}")
        print(f"Validation predictions range: [{np.nanmin(ast_val_probs_aligned):.4f}, {np.nanmax(ast_val_probs_aligned):.4f}]")

    # Optimize weights (optional)
    if OPTIMIZE_WEIGHTS:
        print("\n--- Optimizing Ensemble Weights with Focus on Fall Detection ---")
        prob_list_val = [ast_val_probs_aligned, lgb_val_probs_aligned, rf_val_probs_aligned]
        model_names_val = ["AST", "LGB", "RF"]
        
        # First, evaluate individual model performance on fall detection
        print("\nEvaluating individual model performance on fall detection:")
        model_recalls = {}
        for name, probs in zip(model_names_val, prob_list_val):
            # Find threshold that achieves minimum recall constraint
            precisions, recalls, thresholds = precision_recall_curve(y_val_true_aligned, probs)
            # Note: precision_recall_curve returns arrays where len(precisions) = len(recalls) = len(thresholds) + 1
            # We need to handle this properly when filtering thresholds
            valid_indices = np.where(recalls[:-1] >= MIN_RECALL_CONSTRAINT)[0]  # Exclude last element of recalls
            if len(valid_indices) > 0:
                best_threshold_idx = valid_indices[0]
                best_threshold = thresholds[best_threshold_idx]
                preds = (probs >= best_threshold).astype(int)
                recall = recall_score(y_val_true_aligned, preds)
                precision = precision_score(y_val_true_aligned, preds)
                f1 = f1_score(y_val_true_aligned, preds)
                model_recalls[name] = {
                    'recall': recall,
                    'precision': precision,
                    'f1': f1,
                    'threshold': best_threshold
                }
                print(f"{name} - Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, Threshold: {best_threshold:.4f}")
            else:
                print(f"Warning: {name} could not achieve minimum recall constraint of {MIN_RECALL_CONSTRAINT}")
                model_recalls[name] = {'recall': 0.0, 'precision': 0.0, 'f1': 0.0, 'threshold': 0.5}
        
        # Adjust initial weights based on model performance
        total_recall = sum(model_recalls[name]['recall'] for name in model_names_val)
        if total_recall > 0:
            adjusted_weights = {
                name: model_recalls[name]['recall'] / total_recall 
                for name in model_names_val
            }
            print("\nAdjusted initial weights based on recall performance:")
            for name, weight in adjusted_weights.items():
                print(f"{name}: {weight:.4f}")
        else:
            print("\nUsing default initial weights due to poor recall performance")
            adjusted_weights = ENSEMBLE_WEIGHTS
        
        # Run weight optimization with recall focus
        weight_opt_results = optimize_ensemble_weights(
            y_val_true_aligned, 
            prob_list_val, 
            model_names_val, 
            metric="recall",  # Focus on recall
            step=0.02  # Finer grid search for better recall/precision trade-off
        )
        
        # Combine adjusted weights with optimized weights
        final_weights = {}
        for name in model_names_val:
            # Blend adjusted weights with optimized weights
            final_weights[name] = 0.7 * weight_opt_results["best_weights"][name] + 0.3 * adjusted_weights[name]
        
        # Normalize weights to sum to 1
        total_weight = sum(final_weights.values())
        final_ensemble_weights = {name: weight/total_weight for name, weight in final_weights.items()}
        
        print("\nFinal ensemble weights:")
        for name, weight in final_ensemble_weights.items():
            print(f"{name}: {weight:.4f}")
            
        # Save weight optimization results
        weight_results = {
            'individual_performance': model_recalls,
            'optimized_weights': weight_opt_results["best_weights"],
            'adjusted_weights': adjusted_weights,
            'final_weights': final_ensemble_weights
        }
        
        results_path = os.path.join(RESULTS_DIR, "weight_optimization_results.json")
        with open(results_path, 'w') as f:
            json.dump(weight_results, f, indent=4)
        print(f"\nWeight optimization results saved to: {results_path}")
    else:
        final_ensemble_weights = ENSEMBLE_WEIGHTS
        print(f"Using initial ensemble weights: {final_ensemble_weights}")

    # Calculate final ensemble probabilities on validation set
    ensemble_val_probs = predict_ensemble(ast_val_probs_aligned, lgb_val_probs_aligned, rf_val_probs_aligned, final_ensemble_weights)

    # --- Threshold sweep for recall/precision trade-off ---
    print("\nThreshold sweep (validation set):")
    for thr in np.arange(0.0, 1.01, 0.02):
        preds = (ensemble_val_probs >= thr).astype(int)
        rec = recall_score(y_val_true_aligned, preds)
        prec = precision_score(y_val_true_aligned, preds)
        print(f"Threshold: {thr:.2f} | Recall: {rec:.3f} | Precision: {prec:.3f}")

    # Plot PR vs Threshold for validation set
    plot_precision_recall_vs_threshold(y_val_true_aligned, ensemble_val_probs, title_suffix="_validation")

    # Find optimal threshold based on validation set with recall constraint
    print(f"\nFinding optimal threshold with minimum recall constraint of {MIN_RECALL_CONSTRAINT}")
    optimal_threshold = find_optimal_threshold(
        y_val_true_aligned, 
        ensemble_val_probs, 
        target_metric=OPTIMIZE_THRESHOLD_TARGET, 
        min_recall=MIN_RECALL_CONSTRAINT,
        max_fpr=MAX_FALSE_POSITIVE_RATE
    )
    print(f"Final Optimal Threshold: {optimal_threshold:.4f}")
    # Save the best threshold for reference
    with open(os.path.join(RESULTS_DIR, "best_threshold.txt"), "w") as f:
        f.write(f"Best threshold for recall={MIN_RECALL_CONSTRAINT}: {optimal_threshold:.4f}\n")

    # 5. Final Evaluation on Test Set
    print("\n--- Final Evaluation on Test Set --- ")
    test_file_paths = list(test_df["file_path"])
    y_test_true = np.array(test_df["label"])

    # Get predictions from each model on the test set
    ast_test_probs, ast_valid_indices = predict_with_ast(ast_model_tuned, feature_extractor, test_file_paths, batch_size=AST_BATCH_SIZE)
    lgb_test_probs, rf_test_probs, ml_valid_indices = predict_ml_models(lgb_model, rf_model, scaler, test_file_paths)

    # Find common valid indices between AST and ML predictions
    common_valid_indices = np.intersect1d(ast_valid_indices, ml_valid_indices)
    if len(common_valid_indices) == 0:
        print("Error: No common valid predictions between AST and ML models. Cannot proceed with final evaluation.")
        return

    # Align test predictions using common valid indices
    ast_test_probs_aligned = ast_test_probs[common_valid_indices]
    lgb_test_probs_aligned = lgb_test_probs[common_valid_indices]
    rf_test_probs_aligned = rf_test_probs[common_valid_indices]
    y_test_true_aligned = y_test_true[common_valid_indices]
    test_file_paths_aligned = [test_file_paths[i] for i in common_valid_indices]

    print(f"\nAligned test set size: {len(common_valid_indices)}")
    print(f"AST valid predictions: {len(ast_valid_indices)}")
    print(f"ML valid predictions: {len(ml_valid_indices)}")
    print(f"Common valid predictions: {len(common_valid_indices)}")

    # Calculate final ensemble probabilities on test set
    ensemble_test_probs = predict_ensemble(ast_test_probs_aligned, lgb_test_probs_aligned, rf_test_probs_aligned, final_ensemble_weights)
    ensemble_test_preds = (ensemble_test_probs >= optimal_threshold).astype(int)

    # Calculate and print final metrics
    print("\nFinal Test Set Performance:")
    try:
        report = classification_report(y_test_true_aligned, ensemble_test_preds, target_names=["normal", "fall"])
        print("Classification Report (Test Set):")
        print(report)
        
        # Calculate additional metrics
        tn, fp, fn, tp = confusion_matrix(y_test_true_aligned, ensemble_test_preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        print("\nDetailed Metrics:")
        print(f"False Positive Rate: {fpr:.4f}")
        print(f"Recall (Fall Detection Rate): {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Number of False Negatives (Missed Falls): {fn}")
        print(f"Number of False Positives (False Alarms): {fp}")
        
        # Save report
        with open(os.path.join(RESULTS_DIR, "final_classification_report.txt"), "w") as f:
            f.write(report)
            f.write("\nDetailed Metrics:\n")
            f.write(f"False Positive Rate: {fpr:.4f}\n")
            f.write(f"Recall (Fall Detection Rate): {recall:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Number of False Negatives (Missed Falls): {fn}\n")
            f.write(f"Number of False Positives (False Alarms): {fp}\n")
        
        # Plot and save confusion matrix
        cm = confusion_matrix(y_test_true_aligned, ensemble_test_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Final Confusion Matrix (Test Set - Thr={optimal_threshold:.2f})")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(ticks=[0.5, 1.5], labels=["normal", "fall"])
        plt.yticks(ticks=[0.5, 1.5], labels=["normal", "fall"], rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "final_confusion_matrix.png"))
        plt.close()
        print("Confusion matrix saved.")
        
        # Calculate and print AUC
        auc = roc_auc_score(y_test_true_aligned, ensemble_test_probs)
        print(f"AUC (Test Set): {auc:.4f}")
        
    except Exception as e:
        print(f"Error generating final report/plots: {e}")
        print(f"Unique True Labels: {np.unique(y_test_true_aligned)}")
        print(f"Unique Predicted Labels: {np.unique(ensemble_test_preds)}")

    # After error analysis and targeted augmentation, automatically add new FNs to training set and retrain for higher recall
    # This loop will run twice: initial training, then retraining with new FNs augmentations
    retrain_with_aug = True
    for train_round in range(2):
        print(f"\n=== Training Round {train_round+1} ===")
        # 6. Error Analysis on Test Set
        print("\n--- Running Error Analysis on Test Set Misclassifications ---")
        error_analysis_results = analyze_and_save_errors(
            y_true=y_test_true_aligned,
            y_pred=ensemble_test_preds,
            y_prob=ensemble_test_probs,
            file_paths=test_file_paths_aligned,
            output_dir=ERROR_ANALYSIS_DIR
        )

        # 7. Analyze Error Patterns and Generate Targeted Augmentations
        print("\n--- Analyzing Error Patterns and Generating Targeted Augmentations ---")
        # Load error analysis results
        error_report_path = os.path.join(ERROR_ANALYSIS_DIR, "error_analysis_report.csv")
        if os.path.exists(error_report_path):
            error_df = pd.read_csv(error_report_path)
            false_negatives = error_df[error_df['error_type'] == 'FN']
            if retrain_with_aug and train_round == 0 and len(false_negatives) > 0:
                print("\n[Auto-Retrain] Generating targeted augmentations for missed falls and adding to training set...")
                fn_aug_dir = os.path.join(AUGMENTED_DATA_DIR, "auto_fn_augs")
                os.makedirs(fn_aug_dir, exist_ok=True)
                rir_files = load_rir_files(RIR_DATASET_DIR)
                for _, row in false_negatives.iterrows():
                    file_path = row['file_path']
                    if not os.path.exists(file_path):
                        continue
                    augmented_paths = generate_augmented_audio_for_ml(
                        file_path,
                        fn_aug_dir,
                        rir_files,
                        num_augs=5,  # More augmentations for missed falls
                        time_stretch_range=(0.8, 1.2),
                        pitch_shift_range=(-2, 2),
                        noise_level_range=(0.01, 0.05),
                        rir_probability=0.8,
                        gain_range=(-2, 4)
                    )
                    # Add new augmentations to training set for next round
                    for aug_path in augmented_paths:
                        # Add new augmentations to training set for next round
                        new_row = {
                            "file_path": aug_path,
                            "filename": os.path.basename(aug_path),
                            "fold": 1,  # Ensure it's in training set
                            "subject": -1,
                            "environment": -1,
                            "label": 1,
                            "category": "auto_fn_aug",
                            "confidence": 1.0
                        }
                        train_df = pd.concat([train_df, pd.DataFrame([new_row])], ignore_index=True)
                print(f"[Auto-Retrain] Added {len(false_negatives)*5} new FN augmentations to training set.")
            # After first round, do not retrain again
            if train_round == 1:
                print("[Auto-Retrain] Completed retraining with FN augmentations.")
                break
            # After FN augmentations, also generate targeted augmentations for recent FPs
            if retrain_with_aug and train_round == 0:
                # Load error analysis report
                error_report_path = os.path.join(ERROR_ANALYSIS_DIR, "error_analysis_report.csv")
                if os.path.exists(error_report_path):
                    error_df = pd.read_csv(error_report_path)
                    false_positives = error_df[error_df['error_type'] == 'FP']
                    if len(false_positives) > 0:
                        print("\n[Auto-Retrain] Generating targeted augmentations for recent false positives and adding to training set...")
                        fp_aug_dir = os.path.join(AUGMENTED_DATA_DIR, "auto_fp_augs")
                        os.makedirs(fp_aug_dir, exist_ok=True)
                        rir_files = load_rir_files(RIR_DATASET_DIR)
                        for _, row in false_positives.iterrows():
                            file_path = row['file_path']
                            if not os.path.exists(file_path):
                                continue
                            augmented_paths = generate_augmented_audio_for_ml(
                                file_path,
                                fp_aug_dir,
                                rir_files,
                                num_augs=5,  # More augmentations for hard FPs
                                time_stretch_range=(0.8, 1.2),
                                pitch_shift_range=(-2, 2),
                                noise_level_range=(0.01, 0.05),
                                rir_probability=0.8,
                                gain_range=(-2, 4)
                            )
                            for aug_path in augmented_paths:
                                new_row = {
                                    "file_path": aug_path,
                                    "filename": os.path.basename(aug_path),
                                    "fold": 1,  # Ensure it's in training set
                                    "subject": -1,
                                    "environment": -1,
                                    "label": 0,  # Non-fall
                                    "category": "auto_fp_aug",
                                    "confidence": 1.0
                                }
                                train_df = pd.concat([train_df, pd.DataFrame([new_row])], ignore_index=True)
                        print(f"[Auto-Retrain] Added {len(false_positives)*5} new FP augmentations to training set.")
        else:
            print(f"Error analysis report not found at: {error_report_path}")

        end_time = datetime.now()
        print(f"\n===== Pipeline Finished in: {end_time - start_time} ====")

if __name__ == "__main__":
    main()

