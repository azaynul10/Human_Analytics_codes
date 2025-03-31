import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import skew, kurtosis
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import MobileNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from scikeras.wrappers import KerasClassifier
from tqdm import tqdm  # Import tqdm for progress bars
from lightgbm.callback import early_stopping
from scipy import signal
from sklearn.base import BaseEstimator, ClassifierMixin

# Configuration
DATA_DIR = r"C:\Users\T2430477\Downloads\fan-20250324T081311Z-001\fan\id_00"
NORMAL_DIR = os.path.join(DATA_DIR, 'normal')
ABNORMAL_DIR = os.path.join(DATA_DIR, 'abnormal')
SR = 22050
DURATION = 5
N_MELS = 128
BATCH_SIZE = 128  # Increased from 64 to better utilize GPU
EPOCHS = 100
MODEL_PATH = 'best_fall_detection_model.h5'

os.makedirs('results', exist_ok=True)

# Optimize GPU memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Allow TensorFlow to allocate memory as needed rather than grabbing all at once
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU is available: {gpus}")
        # Set specific GPU to use if you have multiple
        tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(f"GPU error: {e}")
else:
    print("No GPU found. Using CPU instead.")
    tf.config.set_visible_devices([], 'GPU')

# Create TensorFlow dataset with optimized performance
def create_dataset(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    return dataset.cache().shuffle(buffer_size=5000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Fixed Mel spectrogram extraction function
def create_spectrogram(audio_file=None, audio=None, sr=SR):
    try:
        # Load audio file if needed
        if audio is None and audio_file is not None:
            audio, sr = librosa.load(audio_file, sr=sr, res_type='kaiser_fast')

        # Ensure consistent length
        target_length = sr * DURATION
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]

        # Apply pre-emphasis to enhance high frequencies
        audio = librosa.effects.preemphasis(audio, coef=0.97)

        # Create mel spectrogram with consistent parameters
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=N_MELS, n_fft=2048, hop_length=512, fmax=8000
        )

        # Convert to decibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Pad or trim the spectrogram to ensure it has consistent shape
        target_frames = 216
        if mel_spec_db.shape[1] < target_frames:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, target_frames - mel_spec_db.shape[1])), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :target_frames]

        return mel_spec_db  # Return the mel spectrogram in 2D shape directly

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None
def predict_audio(audio_file, models):
    try:
        # Extract features
        specs, feats = process_long_audio(audio_file)

        if len(specs) == 0 or len(feats) == 0:
            print(f"Could not process audio file: {audio_file}")
            return None

        # Get models
        cnn_model = models['cnn_model']
        stacking_model = models['stacking_model']
        scaler = models['scaler']
        base_threshold = models['threshold']
        
        # Make predictions for each segment
        results = []
        
        for spec, feat in zip(specs, feats):
            try:
                # Process spectrogram
                spec_array = np.array([spec])
                if len(spec_array.shape) == 3:
                    spec_array = spec_array.reshape(spec_array.shape[0], spec_array.shape[1], spec_array.shape[2], 1)
                if spec_array.shape[-1] == 1:
                    spec_array = np.repeat(spec_array, 3, axis=-1)
                
                # Ensure fixed dimensions
                target_shape = (1, 128, 216, 3)
                if spec_array.shape != target_shape:
                    resized_spec = np.zeros(target_shape)
                    h = min(spec_array.shape[1], target_shape[1])
                    w = min(spec_array.shape[2], target_shape[2])
                    resized_spec[0, :h, :w, :] = spec_array[0, :h, :w, :]
                    spec_array = resized_spec
                
                # Process features
                feat_scaled = scaler.transform([feat])

                # Get prediction scores
                cnn_pred = float(cnn_model.predict(spec_array, verbose=0)[0][0])
                stack_pred = float(stacking_model.predict_proba(feat_scaled)[0][1])
                
                results.append({
                    'stacking_score': stack_pred,
                    'cnn_score': cnn_pred
                })
            except Exception as e:
                print(f"Error processing segment: {e}")
                continue

        if len(results) == 0:
            return None
            
        # Multi-criteria decision logic
        # 1. Calculate segment statistics
        stack_scores = [r['stacking_score'] for r in results]
        cnn_scores = [r['cnn_score'] for r in results]
        
        max_stack_score = max(stack_scores)
        max_stack_idx = np.argmax(stack_scores)
        avg_stack_score = np.mean(stack_scores)
        
        # 2. Score-based features
        very_high_score = max_stack_score >= 0.85
        high_avg_score = avg_stack_score >= 0.6
        
        # 3. Temporal pattern features
        # Count segments with high scores (potential falls)
        fall_segments = sum(1 for s in stack_scores if s >= base_threshold * 0.8)
        fall_ratio = fall_segments / len(results)
        
        # Check for consecutive high scores
        consecutive_high = 0
        max_consecutive = 0
        for score in stack_scores:
            if score >= base_threshold * 0.7:
                consecutive_high += 1
            else:
                consecutive_high = 0
            max_consecutive = max(max_consecutive, consecutive_high)
        
        # 4. Decision fusion
        # This balances the need to catch all falls (high recall) 
        # with the need to avoid false positives (high precision)
        is_fall = (
            very_high_score or                    # Clear fall signature
            (high_avg_score and fall_ratio > 0.3) or  # Consistent fall pattern
            max_consecutive >= 2 or                # Temporal continuity
            (max_stack_score >= 0.75 and fall_segments >= 2)  # Strong multiple evidence
        )
        
        # If no fall detected, make sure it's not a borderline case
        if not is_fall and max_stack_score >= 0.7:
            # These are harder cases - check CNN score as additional evidence
            if max(cnn_scores) >= 0.6:
                is_fall = True
        
        # Prepare final result with best segment
        result = {
            'stacking_score': max_stack_score,
            'cnn_score': cnn_scores[max_stack_idx],
            'is_fall': is_fall,
            'avg_score': avg_stack_score,
            'fall_ratio': fall_ratio,
            'max_consecutive': max_consecutive
        }
        
        return result

    except Exception as e:
        print(f"Error predicting audio: {e}")
        return None


# Extract fixed-length feature vector for tabular models
def extract_features(audio, sr):
    """Extract advanced audio features with consistent dimensions"""
    try:
        # Ensure consistent length
        target_length = sr * DURATION
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        # Chroma feature (12 pitch classes)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        # Tonnetz (4 features)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        tonnetz_std = np.std(tonnetz, axis=1)

        # Spectral flux (rate of change of the power spectrum)
        spectral_flux = librosa.onset.onset_strength(y=audio, sr=sr)
        spectral_flux_mean = np.mean(spectral_flux)
        spectral_flux_std = np.std(spectral_flux)

        # Scalar features (zero-crossing rate, centroid, bandwidth, etc.)
        zero_crossing = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
        spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        onset_strength = np.mean(onset_env[onset_frames]) if len(onset_frames) > 0 else 0
        onset_count = len(onset_frames)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)

        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_mean = float(np.mean(harmonic**2))
        percussive_mean = float(np.mean(percussive**2))

        # Combine all features
        features = np.concatenate([
            mfccs_mean,  # Shape: (13,)
            mfccs_std,   # Shape: (13,)
            chroma_mean,  # Shape: (12,)
            chroma_std,   # Shape: (12,)
            tonnetz_mean, # Shape: (4,)
            tonnetz_std,  # Shape: (4,)
            [spectral_flux_mean],  # Scalar feature
            [spectral_flux_std],   # Scalar feature
            [zero_crossing],
            [spectral_centroid],
            [spectral_bandwidth],
            [spectral_rolloff],
            [harmonic_mean],
            [percussive_mean],
            [onset_strength],
            [onset_count],
            contrast_mean,
            contrast_std
        ])

        return features

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Optimized audio processing with parallel execution
def prepare_balanced_dataset():
    X_specs = []
    X_feats = []
    y_labels = []

    # Process normal files
    print("\nProcessing normal audio files...")
    normal_files = [f for f in os.listdir(NORMAL_DIR) if f.endswith(('.mp3', '.wav'))]
    print(f"Found {len(normal_files)} normal files")

    # Process abnormal files
    print("\nProcessing abnormal audio files...")
    abnormal_files = [f for f in os.listdir(ABNORMAL_DIR) if f.endswith(('.mp3', '.wav'))]
    print(f"Found {len(abnormal_files)} abnormal files")
    
    # Determine target counts for balancing
    # If we want approximately equal number of samples from each class
    max_samples_per_class = 300  # Target number of samples per class
    
    # Process files with progress tracking
    def process_files(file_paths, label, max_samples):
        results = []
        processed_count = 0
        
        for file_path in tqdm(file_paths, desc=f"Processing {'normal' if label == 0 else 'abnormal'} files"):
            specs, feats = process_long_audio(file_path)
            if not specs or not feats:
                continue
                
            # Limit the number of segments per file to avoid bias from longer files
            max_segments_per_file = 3
            use_segments = min(len(specs), max_segments_per_file)
            
            for i in range(use_segments):
                if processed_count >= max_samples:
                    break
                results.append((specs[i], feats[i], label))
                processed_count += 1
                
            if processed_count >= max_samples:
                break
        
        return results, processed_count
    
    # Process normal files with limit
    normal_results, normal_count = process_files(
        [os.path.join(NORMAL_DIR, f) for f in normal_files],
        label=0,
        max_samples=max_samples_per_class
    )
    
    # Process abnormal files with limit
    abnormal_results, abnormal_count = process_files(
        [os.path.join(ABNORMAL_DIR, f) for f in abnormal_files],
        label=1,
        max_samples=max_samples_per_class
    )
    
    # Combine results
    all_results = normal_results + abnormal_results
    
    # Shuffle to randomize order
    np.random.shuffle(all_results)
    
    # Extract data
    for spec, feat, label in all_results:
        X_specs.append(spec)
        X_feats.append(feat)
        y_labels.append(label)
    
    # Convert to numpy arrays
    if len(X_specs) == 0:
        print("ERROR: No valid audio files could be processed!")
        return np.array([]), np.array([]), np.array([])

    X_specs = np.array(X_specs)
    X_feats = np.array(X_feats)
    y_labels = np.array(y_labels)

    # Check the shape of X_specs before reshaping
    print(f"Shape of X_specs before reshape: {X_specs.shape}")

    # Ensure the spectrogram is in the right shape (128, 216, 1)
    if len(X_specs.shape) == 3:
        X_specs = X_specs.reshape(-1, X_specs.shape[1], X_specs.shape[2], 1)

    # Convert grayscale (1 channel) to RGB (3 channels) by repeating along the last axis
    if X_specs.shape[-1] == 1:
        X_specs = np.repeat(X_specs, 3, axis=-1)

    print(f"\nBalanced dataset prepared:")
    print(f"  Total samples: {len(X_specs)}")
    print(f"  Normal samples: {normal_count}")
    print(f"  Fall samples: {abnormal_count}")
    print(f"  Spectrogram shape: {X_specs.shape}")
    print(f"  Feature vector shape: {X_feats.shape}")
    print(f"  Class distribution: {np.bincount(y_labels)}")

    return X_specs, X_feats, y_labels


# Fixed process_long_audio function
def process_long_audio(audio_file, sr=SR, window_seconds=5, hop_seconds=2.5):
    try:
        # Load audio file
        audio, sr = librosa.load(audio_file, sr=sr)

        # Resample using Scipy instead of Resampy
        target_sr = 22050  # Example target sampling rate
        audio = signal.resample(audio, int(len(audio) * target_sr / sr))
        sr = target_sr

        # Check if audio is too short
        if len(audio) < sr * window_seconds:
            spec = create_spectrogram(audio=audio, sr=sr)
            feat = extract_features(audio, sr)
            if spec is not None and feat is not None:
                return [spec], [feat]
            return [], []

        # Sliding window logic
        window_samples = int(window_seconds * sr)
        hop_samples = int(hop_seconds * sr)

        starts = list(range(0, len(audio) - window_samples + 1, hop_samples))
        specs = []
        feats = []

        # Process each segment sequentially to avoid TensorFlow shape issues
        for start in starts:
            segment = audio[start:start + window_samples]
            spec = create_spectrogram(audio=segment, sr=sr)
            feat = extract_features(segment, sr)
            
            # Only add if processing succeeded
            if spec is not None and feat is not None:
                # Ensure spec has the correct shape
                if spec.shape[0] == N_MELS and spec.shape[1] == 216:
                    specs.append(spec)
                    feats.append(feat)

        return specs, feats
    except Exception as e:
        print(f"Error processing long audio {audio_file}: {e}")
        return [], []


# Optimized CNN model with mixed precision for faster training
def build_robust_cnn_model(input_shape):
    """Build a CNN model with explicit input shapes for better prediction compatibility"""
    # Use the Keras Functional API
    inputs = tf.keras.Input(shape=input_shape, name='input')

    # First branch - standard convolutions
    x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling2D((2, 2))(x1)

    x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    attention1 = layers.Conv2D(1, (1, 1), activation='sigmoid')(x1)
    x1 = layers.Multiply()([x1, attention1])

    x1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.GlobalAveragePooling2D()(x1)

    # Second branch - dilated convolutions for larger receptive field
    x2 = layers.Conv2D(32, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same')(inputs)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    x2 = layers.Conv2D(64, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    x2 = layers.Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.GlobalAveragePooling2D()(x2)

    # Combine branches
    x = layers.Concatenate()([x1, x2])

    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = tf.keras.Model(inputs, outputs)

    # Use Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )

    return model


# Updated CNNWrapper class for stacking
class CNNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, cnn_model, X_specs_data):
        self.cnn_model = cnn_model
        self.X_specs_data = X_specs_data
        self.classes_ = np.array([0, 1])
        
        # Create a mapping between feature indices and spectrogram indices
        self.feature_indices = np.arange(len(X_specs_data))
    
    def fit(self, X, y):
        # Set classes based on training data
        self.classes_ = np.unique(y)
        
        # Create a mapping of indices from X to X_specs_data
        # For simplicity, we'll assume a 1:1 correspondence in the same order
        self.feature_indices = np.arange(min(len(X), len(self.X_specs_data)))
        
        return self
        
    def predict_proba(self, X):
        """Return probability estimates for samples in X"""
        # Create probability array
        probs = np.zeros((X.shape[0], len(self.classes_)))
        
        try:
            # Use the first X.shape[0] spectrograms (or fewer if not enough available)
            n_samples = min(X.shape[0], len(self.X_specs_data))
            specs_to_use = self.X_specs_data[:n_samples]
            
            # Create a batch of spectrograms with correct shape
            batch_specs = np.array(specs_to_use)
            
            # Ensure correct shape for CNN prediction
            if len(batch_specs.shape) == 3:  # (batch, height, width)
                batch_specs = batch_specs.reshape(batch_specs.shape[0], batch_specs.shape[1], batch_specs.shape[2], 1)
            
            # Add RGB channels if needed
            if batch_specs.shape[-1] == 1:
                batch_specs = np.repeat(batch_specs, 3, axis=-1)
            
            # Get predictions in a try-except block to handle TensorFlow errors
            try:
                cnn_preds = self.cnn_model.predict(batch_specs, verbose=0).ravel()
                
                # Fill probability array
                for i in range(n_samples):
                    if i < len(cnn_preds):
                        probs[i, 1] = cnn_preds[i]
                        probs[i, 0] = 1 - cnn_preds[i]
            except Exception as e:
                print(f"Error in CNN prediction: {e}")
                # Fall back to default probabilities (0.5/0.5)
                probs[:n_samples, 1] = 0.5
                probs[:n_samples, 0] = 0.5
        
        except Exception as e:
            print(f"Error in predict_proba: {e}")
            # Fall back to default probabilities
            probs[:, 1] = 0.5
            probs[:, 0] = 0.5
            
        return probs
    
    def predict(self, X):
        """Return class predictions for samples in X"""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {"cnn_model": self.cnn_model, "X_specs_data": self.X_specs_data}
    
    def set_params(self, **parameters):
        """Set parameters for this estimator"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def score(self, X, y):
        """Return accuracy score on given data"""
        return np.mean(self.predict(X) == y)


# Updated train_ensemble_model function with class balancing
def train_ensemble_model(X_specs, X_feats, y_labels):
    # Ensure data shapes are consistent
    if X_specs.shape[0] != X_feats.shape[0]:
        print(f"Warning: Inconsistent sample counts. X_specs: {X_specs.shape[0]}, X_feats: {X_feats.shape[0]}")
        # Take the minimum length to ensure alignment
        min_length = min(X_specs.shape[0], X_feats.shape[0])
        X_specs = X_specs[:min_length]
        X_feats = X_feats[:min_length]
        y_labels = y_labels[:min_length]
        print(f"Adjusted to {min_length} samples")
    
    # Split data with stratification
    X_train_specs, X_test_specs, X_train_feats, X_test_feats, y_train, y_test = train_test_split(
        X_specs, X_feats, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )

    # Create TensorFlow datasets for efficient GPU feeding
    train_dataset = create_dataset(X_train_specs, y_train, BATCH_SIZE)
    test_dataset = create_dataset(X_test_specs, y_test, BATCH_SIZE)
    
    # Calculate class weights for handling imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights[0] = class_weights[0] * 1.75 
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"Class weights: {class_weights_dict}")
    
    # Train CNN model
    print("\nTraining CNN model...")
    cnn_model = build_robust_cnn_model(X_train_specs.shape[1:])
    
    # Get callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=15,
        restore_best_weights=True,
        mode='max'
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor='val_auc',
        save_best_only=True,
        mode='max'
    )
    
    # Reduce LR on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train with dataset API and class weights
    history = cnn_model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        class_weight=class_weights_dict,
        verbose=1
    )
    
    # Prepare data for tabular models
    scaler = StandardScaler()
    X_train_feats_scaled = scaler.fit_transform(X_train_feats)
    X_test_feats_scaled = scaler.transform(X_test_feats)
    
    # Train LightGBM with optimized parameters and class balancing
    print("\nTraining LightGBM model...")
    lgb_params = {
        'num_leaves': 31,
        'max_depth': 5,
        'min_data_in_leaf': 20,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'objective': 'binary',
        'class_weight': 'balanced',  # Handle class imbalance
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'boosting': 'gbdt',
        'random_state': 42
    }
    
    lgb_model = lgb.LGBMClassifier(verbose=0, **lgb_params)
    lgb_model.fit(
        X_train_feats_scaled, 
        y_train,
        eval_set=[(X_test_feats_scaled, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=100)]
    )
    
    # Train RandomForest with parallel processing and class balancing
    print("\nTraining RandomForest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        max_features='sqrt',
        n_jobs=16,
        class_weight='balanced',  # Handle class imbalance
        random_state=42
    )
    rf_model.fit(X_train_feats_scaled, y_train)
    
    # Create improved CNNWrapper for stacking
    cnn_wrapper = CNNWrapper(cnn_model=cnn_model, X_specs_data=X_test_specs)
    
    # Implement stacking
    print("\nImplementing Stacking Ensemble...")
    base_learners = [
        ('cnn', cnn_wrapper),
        ('lgb', lgb_model),
        ('rf', rf_model)
    ]
    
    meta_model = LogisticRegression(
        C=1.0,
        class_weight='balanced',  # Handle class imbalance
        max_iter=200,
        random_state=42
    )
    
    stacking_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_model,
        cv=5,
        n_jobs=16,
        passthrough=False,
        verbose=1
    )
    
    # Train stacking model
    print("Training stacking model...")
    stacking_model.fit(X_train_feats_scaled, y_train)
    
    # Get predictions for evaluation
    cnn_preds = cnn_model.predict(X_test_specs, verbose=0).ravel()
    lgb_preds = lgb_model.predict_proba(X_test_feats_scaled)[:, 1]
    rf_preds = rf_model.predict_proba(X_test_feats_scaled)[:, 1]
    stack_preds = stacking_model.predict_proba(X_test_feats_scaled)[:, 1]
    
    # Find optimal threshold for each model
    def get_optimal_threshold(y_true, y_probs):
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        f1_scores = ((1 + 0.5**2) * precision * recall) / ((0.5**2 * precision) + recall + 1e-7)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5
    
    cnn_threshold = get_optimal_threshold(y_test, cnn_preds)
    lgb_threshold = get_optimal_threshold(y_test, lgb_preds)
    rf_threshold = get_optimal_threshold(y_test, rf_preds)
    stack_threshold = get_optimal_threshold(y_test, stack_preds)
    
    # Apply thresholds
    cnn_binary = (cnn_preds >= cnn_threshold).astype(int)
    lgb_binary = (lgb_preds >= lgb_threshold).astype(int)
    rf_binary = (rf_preds >= rf_threshold).astype(int)
    stack_binary = (stack_preds >= stack_threshold).astype(int)
    
    # Print results for each model
    print("\nCNN Model Results:")
    print(classification_report(y_test, cnn_binary))
    
    print("\nLightGBM Model Results:")
    print(classification_report(y_test, lgb_binary))
    
    print("\nRandomForest Model Results:")
    print(classification_report(y_test, rf_binary))
    
    print("\nStacking Model Results:")
    print(classification_report(y_test, stack_binary))
    
    # Calculate and print metrics for stacking model
    accuracy = accuracy_score(y_test, stack_binary)
    precision = precision_score(y_test, stack_binary)
    recall = recall_score(y_test, stack_binary)
    f1 = f1_score(y_test, stack_binary)
    auc = roc_auc_score(y_test, stack_preds)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Optimal Threshold: {stack_threshold:.4f}")
    
    # Plot confusion matrix for stacking model
    cm = confusion_matrix(y_test, stack_binary)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Stacking Model Confusion Matrix')
    plt.colorbar()
    classes = ['Normal', 'Fall']
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                    horizontalalignment='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('results/stacking_confusion_matrix.png')
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curves
    from sklearn.metrics import roc_curve
    
    # CNN
    fpr_cnn, tpr_cnn, _ = roc_curve(y_test, cnn_preds)
    plt.plot(fpr_cnn, tpr_cnn, label=f'CNN (AUC = {roc_auc_score(y_test, cnn_preds):.3f})')
    
    # LightGBM
    fpr_lgb, tpr_lgb, _ = roc_curve(y_test, lgb_preds)
    plt.plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC = {roc_auc_score(y_test, lgb_preds):.3f})')
    
    # RandomForest
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_preds)
    plt.plot(fpr_rf, tpr_rf, label=f'RandomForest (AUC = {roc_auc_score(y_test, rf_preds):.3f})')
    
    # Stacking
    fpr_stack, tpr_stack, _ = roc_curve(y_test, stack_preds)
    plt.plot(fpr_stack, tpr_stack, label=f'Stacking (AUC = {auc:.3f})')
    
    # Plot settings
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Fall Detection Models')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/roc_curves.png')
    
    return {
        'cnn_model': cnn_model,
        'lgb_model': lgb_model,
        'rf_model': rf_model,
        'stacking_model': stacking_model,
        'scaler': scaler,
        'threshold': stack_threshold,
        'cnn_threshold': cnn_threshold,
        'lgb_threshold': lgb_threshold,
        'rf_threshold': rf_threshold
    }


# Updated main function
def main():
    print("Enhanced Audio-based Fall Detection System with Ensemble Learning")
    print("==============================================================")

    # Prepare balanced dataset
    print("\nStep 1: Preparing balanced dataset...")
    X_specs, X_feats, y_labels = prepare_balanced_dataset()

    # Check if dataset preparation succeeded
    if X_specs.size == 0:
        print("ERROR: Dataset preparation failed.")
        return

    # Train model
    print("\nStep 2: Training ensemble model...")
    models = train_ensemble_model(X_specs, X_feats, y_labels)

    if models is None:
        print("ERROR: Model training failed.")
        return

    # Save the trained models
    print("\nSaving models...")
    os.makedirs('models', exist_ok=True)
    
    # Save the CNN model
    models['cnn_model'].save('models/cnn_model.h5')
    
    # Save other models using joblib
    import joblib
    joblib.dump(models['lgb_model'], 'models/lgb_model.pkl')
    joblib.dump(models['rf_model'], 'models/rf_model.pkl')
    joblib.dump(models['stacking_model'], 'models/stacking_model.pkl')
    joblib.dump(models['scaler'], 'models/scaler.pkl')
    
    # Save thresholds
    with open('models/thresholds.txt', 'w') as f:
        f.write(f"CNN Threshold: {models['cnn_threshold']}\n")
        f.write(f"LightGBM Threshold: {models['lgb_threshold']}\n")
        f.write(f"RandomForest Threshold: {models['rf_threshold']}\n")
        f.write(f"Stacking Threshold: {models['threshold']}\n")
    
    # Test with sample files
    print("\nStep 3: Testing model on sample files...")

    normal_files = [os.path.join(NORMAL_DIR, f) for f in os.listdir(NORMAL_DIR)
                  if f.endswith(('.mp3', '.wav'))]
    abnormal_files = [os.path.join(ABNORMAL_DIR, f) for f in os.listdir(ABNORMAL_DIR)
                    if f.endswith(('.mp3', '.wav'))]

    # For cleaner output, limit the number of test files
    max_test_files = 3
    
    # Test normal files
    test_normal_files = normal_files[:min(len(normal_files), max_test_files)]
    for test_file in test_normal_files:
        print(f"\nTesting normal file: {os.path.basename(test_file)}")
        try:
            result = predict_audio(test_file, models)
            if result:
                print(f"Prediction: {'Fall' if result['is_fall'] else 'Normal'}")
                print(f"Confidence: {result['stacking_score']:.4f}")
                print(f"CNN Score: {result['cnn_score']:.4f}")
            else:
                print("Failed to process file")
        except Exception as e:
            print(f"Error testing file: {e}")
            # Test abnormal files
    test_abnormal_files = abnormal_files[:min(len(abnormal_files), max_test_files)]
    for test_file in test_abnormal_files:
        print(f"\nTesting fall file: {os.path.basename(test_file)}")
        try:
            result = predict_audio(test_file, models)
            if result:
                print(f"Prediction: {'Fall' if result['is_fall'] else 'Normal'}")
                print(f"Confidence: {result['stacking_score']:.4f}")
                print(f"CNN Score: {result['cnn_score']:.4f}")
            else:
                print("Failed to process file")
        except Exception as e:
            print(f"Error testing file: {e}")

    # Plot learning history if available
    if hasattr(models['cnn_model'], 'history') and models['cnn_model'].history is not None:
        history = models['cnn_model'].history.history
        
        # Plot accuracy and loss
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/training_history.png')
    
    print("\nEnhanced fall detection system complete!")
    print("Results and plots have been saved to the 'results' folder.")
    print("Trained models have been saved to the 'models' folder.")
    
    # Create a comprehensive test function
    def comprehensive_test(file_path, expected_label, models):
        print(f"\nTesting file: {os.path.basename(file_path)}")
        print(f"Expected class: {'Fall' if expected_label == 1 else 'Normal'}")
        
        # Analyze audio properties
        try:
            audio, sr = librosa.load(file_path, sr=SR)
            duration = librosa.get_duration(y=audio, sr=sr)
            rms = np.sqrt(np.mean(audio**2))
            snr = np.mean(audio**2) / np.std(audio)
            
            print(f"Audio properties:")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  Signal strength (RMS): {rms:.6f}")
            print(f"  Signal-to-noise estimate: {snr:.2f}")
        except Exception as e:
            print(f"Error analyzing audio: {e}")
        
        # Get prediction
        try:
            result = predict_audio(file_path, models)
            
            if result:
                prediction = 1 if result['is_fall'] else 0
                correct = prediction == expected_label
                
                print(f"Prediction: {'Fall' if result['is_fall'] else 'Normal'}")
                print(f"Confidence: {result['stacking_score']:.4f}")
                print(f"CNN Score: {result['cnn_score']:.4f}")
                print(f"Correct: {'✓' if correct else '✗'}")
                
                return {
                    'file': os.path.basename(file_path),
                    'expected': expected_label,
                    'predicted': prediction,
                    'confidence': result['stacking_score'],
                    'correct': correct
                }
            else:
                print("Failed to process audio file")
                return None
        except Exception as e:
            print(f"Error predicting: {e}")
            return None
    
    # Select test files
    print("\nRunning comprehensive evaluation...")
    
    # Get a representative sample from each class
    np.random.seed(42)  # For reproducibility
    max_eval_files = 10  # Number of files to test per class
    
    # Select test files
    if len(normal_files) > max_eval_files:
        normal_test_files = np.random.choice(normal_files, max_eval_files, replace=False)
    else:
        normal_test_files = normal_files
    
    if len(abnormal_files) > max_eval_files:
        abnormal_test_files = np.random.choice(abnormal_files, max_eval_files, replace=False)
    else:
        abnormal_test_files = abnormal_files
    
    # Run evaluation
    test_results = []
    
    # Test normal files
    for file in normal_test_files:
        result = comprehensive_test(file, 0, models)
        if result:
            test_results.append(result)
    
    # Test abnormal files
    for file in abnormal_test_files:
        result = comprehensive_test(file, 1, models)
        if result:
            test_results.append(result)
    
    # Summarize test results
    if test_results:
        correct_count = sum(1 for r in test_results if r['correct'])
        total_count = len(test_results)
        accuracy = correct_count / total_count * 100
        
        print(f"\nTest Results Summary:")
        print(f"  Total test files: {total_count}")
        print(f"  Correctly classified: {correct_count}")
        print(f"  Test accuracy: {accuracy:.2f}%")
        
        # Calculate per-class metrics
        normal_correct = sum(1 for r in test_results if r['expected'] == 0 and r['correct'])
        normal_total = sum(1 for r in test_results if r['expected'] == 0)
        
        fall_correct = sum(1 for r in test_results if r['expected'] == 1 and r['correct'])
        fall_total = sum(1 for r in test_results if r['expected'] == 1)
        
        if normal_total > 0:
            print(f"  Normal class accuracy: {normal_correct/normal_total*100:.2f}%")
        
        if fall_total > 0:
            print(f"  Fall class accuracy: {fall_correct/fall_total*100:.2f}%")
        
        # Create a test results table
        results_df = pd.DataFrame(test_results)
        results_df['correct'] = results_df['correct'].astype(str)
        results_df.to_csv('results/test_results.csv', index=False)
        print(f"\nDetailed test results saved to 'results/test_results.csv'")
        
        # Create a confusion matrix from test results
        from sklearn.metrics import confusion_matrix
        
        y_true = [r['expected'] for r in test_results]
        y_pred = [r['predicted'] for r in test_results]
        
        classes = ['Normal', 'Fall']

        test_cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(test_cm, interpolation='nearest', cmap='Blues')
        plt.title('Test Set Confusion Matrix')
        plt.colorbar()
        plt.xticks([0, 1], classes)
        plt.yticks([0, 1], classes)
        
        # Add text annotations
        for i in range(test_cm.shape[0]):
            for j in range(test_cm.shape[1]):
                plt.text(j, i, str(test_cm[i, j]),
                        horizontalalignment='center',
                        color='white' if test_cm[i, j] > test_cm.max() / 2 else 'black')
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('results/test_confusion_matrix.png')
    
    print("\nEnhanced fall detection system evaluation complete!")
    print("All results and plots have been saved to the 'results' folder.")
    # Add to your main function
    print("\nAnalyzing decision boundaries:")
    for file in normal_test_files[:3]:
        specs, feats = process_long_audio(file)
        if specs and feats:
            print(f"\nNormal file: {os.path.basename(file)}")
            print("Segment scores:")
            for i, (spec, feat) in enumerate(zip(specs[:5], feats[:5])):  # First 5 segments
                spec_array = np.array([spec])
                if len(spec_array.shape) == 3:
                    spec_array = spec_array.reshape(spec_array.shape[0], spec_array.shape[1], spec_array.shape[2], 1)
                if spec_array.shape[-1] == 1:
                    spec_array = np.repeat(spec_array, 3, axis=-1)
                
                feat_scaled = models['scaler'].transform([feat])
                cnn_pred = float(models['cnn_model'].predict(spec_array, verbose=0)[0][0])
                stack_pred = float(models['stacking_model'].predict_proba(feat_scaled)[0][1])
                
                print(f"  Segment {i+1}: CNN={cnn_pred:.4f}, Stack={stack_pred:.4f}")

    for file in abnormal_test_files[:3]:
        specs, feats = process_long_audio(file)
        if specs and feats:
            print(f"\nFall file: {os.path.basename(file)}")
            print("Segment scores:")
            for i, (spec, feat) in enumerate(zip(specs[:5], feats[:5])):  # First 5 segments
                spec_array = np.array([spec])
                if len(spec_array.shape) == 3:
                    spec_array = spec_array.reshape(spec_array.shape[0], spec_array.shape[1], spec_array.shape[2], 1)
                if spec_array.shape[-1] == 1:
                    spec_array = np.repeat(spec_array, 3, axis=-1)
                
                feat_scaled = models['scaler'].transform([feat])
                cnn_pred = float(models['cnn_model'].predict(spec_array, verbose=0)[0][0])
                stack_pred = float(models['stacking_model'].predict_proba(feat_scaled)[0][1])
                
                print(f"  Segment {i+1}: CNN={cnn_pred:.4f}, Stack={stack_pred:.4f}")
    results_df = pd.DataFrame(test_results)
    full_path = os.path.join(os.getcwd(), 'results/test_results.csv')
    results_df.to_csv(full_path, index=False)
    print(f"Saved results to: {full_path}")
    print("\nDetailed test results:")
    for result in test_results:
        print(f"File: {result['file']}, Expected: {'Fall' if result['expected'] == 1 else 'Normal'}, " +
            f"Predicted: {'Fall' if result['predicted'] == 1 else 'Normal'}, " +
            f"Confidence: {result['confidence']:.4f}, " +
            f"Correct: {result['correct']}")

# Create an application to perform real-time prediction
def create_prediction_app():
    # Check if models exist
    if not os.path.exists('models/cnn_model.h5'):
        print("Error: Models not found. Please train the models first.")
        return
    
    print("\nLoading trained models for prediction...")
    import joblib
    
    # Load models
    cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
    lgb_model = joblib.load('models/lgb_model.pkl')
    rf_model = joblib.load('models/rf_model.pkl')
    stacking_model = joblib.load('models/stacking_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Load thresholds
    threshold = 0.5  # Default
    with open('models/thresholds.txt', 'r') as f:
        for line in f:
            if line.startswith('Stacking Threshold:'):
                threshold = float(line.split(':')[1].strip())
    
    # Create model dictionary
    models = {
        'cnn_model': cnn_model,
        'lgb_model': lgb_model,
        'rf_model': rf_model,
        'stacking_model': stacking_model,
        'scaler': scaler,
        'threshold': threshold
    }
    
    # Function to predict a single file
    def predict_file(file_path):
        try:
            result = predict_audio(file_path, models)
            if result:
                prediction = 'Fall' if result['is_fall'] else 'Normal'
                confidence = result['stacking_score']
                return prediction, confidence
            else:
                return "Error", 0.0
        except Exception as e:
            print(f"Error predicting file: {e}")
            return "Error", 0.0
    
    # Simple command-line interface for prediction
    while True:
        print("\nFall Detection Prediction App")
        print("Enter 'q' to quit")
        file_path = input("Enter audio file path: ")
        
        if file_path.lower() == 'q':
            break
        
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            continue
        
        print(f"\nAnalyzing file: {os.path.basename(file_path)}")
        prediction, confidence = predict_file(file_path)
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()
   