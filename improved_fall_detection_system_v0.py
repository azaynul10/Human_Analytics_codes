import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import tensorflow as tf
import seaborn as sns
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, log_loss, brier_score_loss
from tensorflow.keras import layers, models, callbacks
import visualkeras
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.utils.class_weight import compute_class_weight
warnings.filterwarnings("ignore", message=".*resource_tracker.*")
warnings.filterwarnings("ignore", message=".*legend_text_spacing_offset.*")
warnings.filterwarnings("ignore", message=".*step.*is already reported.*")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
import pickle
from concurrent.futures import ThreadPoolExecutor
import optuna
from optuna.integration import TFKerasPruningCallback
from sklearn.model_selection import cross_val_score
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import resample_poly
from sklearn.calibration import calibration_curve
from scipy.special import expit
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
# Set this to True to enable extensive testing and plotting
RUN_EXTENSIVE_TEST = True

DATA_DIR = r'C:\Users\Abedi\raw\audio'
NORMAL_DIR = os.path.join(DATA_DIR, 'normal')
ABNORMAL_DIR = os.path.join(DATA_DIR, 'abnormal')
SR = 22050
DURATION = 5
N_MELS = 128
BATCH_SIZE = 8
SEED = 42 

# Set seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs('results', exist_ok=True)

def sanitize_features(features, dtype=np.float32):
    features = features.astype(dtype)
    eps = 1e-8
    features = np.nan_to_num(features, nan=0.0, posinf=np.finfo(dtype).max, neginf=np.finfo(dtype).min)
    features = np.clip(features, -1e5, 1e5)
    return features

def extract_features(audio, sr):
    try:
        if len(audio) > sr * DURATION:
            audio = audio[:sr * DURATION]
        elif len(audio) < sr * DURATION:
            padding = np.zeros(sr * DURATION - len(audio))
            audio = np.concatenate([audio, padding])

        rms = np.sqrt(np.mean(np.square(audio)))
        zero_crossing_rate = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))

        stft = np.abs(librosa.stft(audio))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr), axis=1)
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_var = np.var(mfccs, axis=1)

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel = np.maximum(mel, 1e-8)  # avoid log(0)
        mel_mean = np.mean(mel, axis=1)
        mel_var = np.var(mel, axis=1)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_var = np.var(chroma, axis=1)

        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        tonnetz_var = np.var(tonnetz, axis=1)

        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onset_mean = np.mean(onset_env)
        onset_var = np.var(onset_env)

        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        tempo_mean = np.mean(tempogram, axis=1)
        tempo_var = np.var(tempogram, axis=1)

        features = np.concatenate([
            [rms, zero_crossing_rate, spec_cent, spec_bw, rolloff, zcr, onset_mean, onset_var],
            mfccs_mean, mfccs_var,
            mel_mean[:50], mel_var[:50],
            chroma_mean, chroma_var,
            tonnetz_mean, tonnetz_var,
            tempo_mean[:20], tempo_var[:20]
        ])

        return sanitize_features(features)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None
def build_vggish(input_shape=(96, 64, 1)):
    model = models.Sequential(name="VGGish_Style_CNN")
    
    # Block 1
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 2
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 3
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 4
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.GlobalAveragePooling2D())

    # Dense Layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



def build_i3d_stub(input_shape=(32, 64, 64, 1)):
    input_layer = layers.Input(shape=input_shape)

    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling3D((1, 2, 2), strides=(1, 2, 2), padding='same')(x)

    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer, name="I3D_Style_Model")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
def create_spectrogram(audio, sr):
    """Generate mel spectrogram for CNN input"""
    try:
        # Ensure audio is the right length
        if len(audio) > sr * DURATION:
            audio = audio[:sr * DURATION]
        elif len(audio) < sr * DURATION:
            # Pad with zeros if too short
            padding = np.zeros(sr * DURATION - len(audio))
            audio = np.concatenate([audio, padding])
        
        # Generate stereo-like representation (2 channels)
        # Channel 1: Regular mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr,
            n_mels=N_MELS,
            n_fft=2048,
            hop_length=512,
            fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Channel 2: Delta (derivative) of mel spectrogram to capture changes
        mel_spec_delta = librosa.feature.delta(mel_spec_db)
        
        spec_data = np.stack([mel_spec_db, mel_spec_delta])
        spec_data = np.expand_dims(spec_data, axis=-1)
        
        return spec_data
    except Exception as e:
        print(f"Error generating spectrogram: {e}")
        return None


def is_valid_audio_file(file_path):
    """Check if an audio file is valid and can be processed"""
    try:
        audio, sr = librosa.load(file_path, sr=SR, duration=0.1)

        if np.isnan(audio).any() or np.isinf(audio).any():
            return False

        if np.mean(np.abs(audio)) < 1e-6:
            return False

        return True
    except Exception:
        return False

def load_audio_file(file_path, target_sr=16000):
    # 1. Load audio data using soundfile or pydub
    try:
        audio_data, sr = sf.read(file_path, dtype='float32')  
    except Exception:
        try:
            audio = AudioSegment.from_file(file_path)
            sr = audio.frame_rate
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels > 1:
                samples = samples.reshape((-1, audio.channels))
                samples = samples.mean(axis=1)  
            samples /= (1 << (8 * audio.sample_width - 1))  # e.g. 16-bit audio -> divide by 32768
            audio_data = samples
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None, None
    
    if sr != target_sr:
        # polyphase resampling for quality
        audio_data = resample_poly(audio_data, up=target_sr, down=sr)
        sr = target_sr
    
    if audio_data.ndim == 2 and audio_data.shape[1] > 1:
        audio_data = audio_data.mean(axis=1)  
    audio_data = np.asarray(audio_data, dtype=np.float32)  
    if audio_data.size == 0:
        print(f"Loaded audio is empty: {file_path}")
        return None, None
    
    if not np.all(np.isfinite(audio_data)):
        print(f"Audio data has NaN or Inf values: {file_path}")
        return None, None
    
    #  check for non-silent audio (simple energy check)
    if np.max(np.abs(audio_data)) < 1e-5:
        print(f"Warning: {file_path} may be silent or low-volume.")
    
    return audio_data, sr

def prepare_dataset():
    cache_dir = os.path.join(os.path.dirname(DATA_DIR), 'cache')
    os.makedirs(os.path.join(cache_dir, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, 'abnormal'), exist_ok=True)
    
    X_specs = []  # Spectrograms for CNN
    X_feats = []  # Feature vectors for LightGBM
    y_labels = []  # Labels
    file_paths = []  # File paths for reference

    print("\nProcessing normal audio files...")
    normal_files = [f for f in os.listdir(NORMAL_DIR) if f.endswith(('.mp3', '.wav'))]
    print(f"Found {len(normal_files)} normal files")

    for i, file in enumerate(normal_files):
        if i % 10 == 0:
            print(f"Processing normal file {i+1}/{len(normal_files)}")
        global file_path
        file_path = os.path.join(NORMAL_DIR, file)
        cache_file = os.path.join(cache_dir, 'normal', file + '.pkl')
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    X_specs.append(cached_data['spec'])
                    X_feats.append(cached_data['feat'])
                    y_labels.append(0)
                    file_paths.append(file_path)
                print(f"Cache hit for {file}")
                continue
            except Exception as e:
                print(f"Cache error for {file}: {e}, reprocessing...")
                
        # Process if not cached
        try:
            audio, sr = load_audio_file(file_path, target_sr=SR)
            if audio is None:
                print(f"Skipping {file} - loading failed")
                continue
                
            spec = create_spectrogram(audio=audio, sr=sr)
            feat = extract_features(audio, sr)
            
            if spec is not None and feat is not None:
                # Cache the results
                with open(cache_file, 'wb') as f:
                    pickle.dump({'spec': spec, 'feat': feat}, f)
                    
                X_specs.append(spec)
                X_feats.append(feat)
                y_labels.append(0)  # 0 for normal
                file_paths.append(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("\nProcessing abnormal audio files...")
    abnormal_files = [f for f in os.listdir(ABNORMAL_DIR) if f.endswith(('.mp3', '.wav'))]
    print(f"Found {len(abnormal_files)} abnormal files")

    for i, file in enumerate(abnormal_files):
        if i % 10 == 0:
            print(f"Processing abnormal file {i+1}/{len(abnormal_files)}")

        file_path = os.path.join(ABNORMAL_DIR, file)
        cache_file = os.path.join(cache_dir, 'abnormal', file + '.pkl')
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    X_specs.append(cached_data['spec'])
                    X_feats.append(cached_data['feat'])
                    y_labels.append(1)
                    file_paths.append(file_path)
                print(f"Cache hit for {file}")
                continue
            except Exception as e:
                print(f"Cache error for {file}: {e}, reprocessing...")
        
        # Process if not cached
        try:
            audio, sr = load_audio_file(file_path, target_sr=SR)
            if audio is None:
                print(f"Skipping {file} - loading failed")
                continue
                
            spec = create_spectrogram(audio=audio, sr=sr)
            feat = extract_features(audio, sr)
            
            if spec is not None and feat is not None:
                with open(cache_file, 'wb') as f:
                    pickle.dump({'spec': spec, 'feat': feat}, f)
                    
                X_specs.append(spec)
                X_feats.append(feat)
                y_labels.append(1)  # 1 for abnormal/fall
                file_paths.append(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if len(X_specs) == 0:
        print("ERROR: No valid audio files could be processed!")
        return np.array([]), np.array([]), np.array([]), []

    X_specs = np.array(X_specs)
    X_feats = np.array(X_feats)
    y_labels = np.array(y_labels)
    X_specs = X_specs.astype(np.float32)
    X_feats = X_feats.astype(np.float32)
    print(f"\nDataset prepared:")
    print(f"  Total samples: {len(X_specs)}")
    print(f"  Spectrogram shape: {X_specs.shape}")
    print(f"  Feature vector shape: {X_feats.shape}")
    print(f"  Class distribution: {np.bincount(y_labels)}")

    return X_specs, X_feats, y_labels, file_paths
def build_vggish(input_shape=(96, 64, 1)):
    model = models.Sequential(name="VGGish_Style_CNN")
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def build_simple_cnn(input_shape, dropout_rate=0.5, l2_reg=0.001):
    """Build a more expressive CNN model with stronger regularization"""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Input normalization
    x = tf.keras.layers.BatchNormalization()(inputs)
    
    # Multi-scale feature extraction with same pooling dimensions
    branch1 = tf.keras.layers.Conv3D(32, (1, 3, 3), activation='relu', padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)  
    branch1 = tf.keras.layers.MaxPooling3D((1, 2, 2), padding='same')(branch1)
    branch1 = tf.keras.layers.BatchNormalization()(branch1)

    branch2 = tf.keras.layers.Conv3D(32, (1, 5, 5), activation='relu', padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    branch2 = tf.keras.layers.MaxPooling3D((1, 2, 2), padding='same')(branch2)
    branch2 = tf.keras.layers.BatchNormalization()(branch2)
    
    merged = tf.keras.layers.Concatenate(axis=-1)([branch1, branch2])
    
    # Attention mechanism 
    attention = tf.keras.layers.Conv3D(64, (1, 1, 1), activation='sigmoid')(merged)
    x = tf.keras.layers.Multiply()([merged, attention])
    
    # Deeper feature processing
    x = tf.keras.layers.Conv3D(64, (1, 3, 3), activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.MaxPooling3D((1, 2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Conv3D(128, (1, 3, 3), activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    
    # Classification head with stronger regularization
    x = tf.keras.layers.Dense(128, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(l2_reg*2))(x)  
    x = tf.keras.layers.Dropout(dropout_rate+0.1)(x)  
    x = tf.keras.layers.Dense(64, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(l2_reg*2))(x)
    x = tf.keras.layers.Dropout(dropout_rate+0.1)(x)
    
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    try:
        import visualkeras
        visualkeras.layered_view(model, 
                            color_map={
                                'Conv3D': 'red',
                                'MaxPooling3D': 'blue',
                                'Dense': 'green',
                                'Dropout': 'yellow',
                                'Reshape': 'purple',
                                'Flatten': 'orange'
                            },
                            to_file='results/architecture.png',
                            legend=True, draw_volume=True, 
                            spacing=50)
    except Exception as e:
        print(f"Visualization skipped - visualkeras error: {e}")
    i3d_model = build_i3d_stub()
    vggish_model = build_vggish()
    visualkeras.layered_view(vggish_model, to_file="vggish_architecture.png", legend=True, draw_volume=True)
    visualkeras.layered_view(i3d_model, to_file="i3d_architecture.png", legend=True, draw_volume=True)
    i3d_model.summary()
    plot_model(model, to_file='results/model_full_architecture.png', show_shapes=True, show_layer_names=True)
    return model
class MetricChecker(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print("Available metrics:", self.model.metrics_names)
def find_optimal_threshold(y_true, y_pred_proba, min_recall=0.98):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    precision = precision[:-1]
    recall = recall[:-1]

    valid_indices = recall >= min_recall

    if not np.any(valid_indices):
        print(f"No threshold achieves {min_recall*100:.1f}% recall, using lowest threshold")
        return thresholds[0]
    if len(thresholds) + 1 == len(recall):
        precision = precision[:-1]
        recall = recall[:-1]

    valid_precision = precision[valid_indices]
    valid_recall = recall[valid_indices]
    valid_thresholds = thresholds[valid_indices]

    beta = 2
    f2_scores = ((1 + beta**2) * valid_precision * valid_recall) / ((beta**2 * valid_precision) + valid_recall + 1e-10)
    best_idx = np.argmax(f2_scores)

    return valid_thresholds[best_idx]

class TemperatureScaling(tf.keras.Model):
    """Temperature scaling layer for probability calibration"""
    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.temperature = tf.Variable(1.0, 
                                      trainable=True, 
                                      constraint=tf.keras.constraints.NonNeg(),
                                      name='temperature')
    
    def call(self, inputs):
        logits = self.base_model(inputs)
        scaled_logits = logits / self.temperature
        return tf.keras.activations.sigmoid(scaled_logits)
    
    def get_config(self):
        return {"base_model": self.base_model}
    
    @classmethod
    def from_config(cls, config):
        return cls(config["base_model"])

def calibrate_model_with_temperature(model, X_val, y_val, epochs=100):
    """Calibrate model predictions using temperature scaling"""
    temp_model = TemperatureScaling(model)
    temp_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy'
    )
    
    temp_model.fit(X_val, y_val, epochs=epochs, verbose=0)
    
    temperature = temp_model.temperature.numpy()
    print(f"Calibration temperature: {temperature:.4f}")
    
    return temp_model
def optimize_ensemble_weights(models, X_val, y_val, metric='f2'):  
    cnn_preds = models['cnn_model'].predict(X_val[0]).flatten()
    lgb_preds = models['lgb_model'].predict_proba(X_val[1])[:, 1]
    rf_preds = models['rf_model'].predict_proba(X_val[2])[:, 1]
    
    best_score = 0
    best_weights = (0.33, 0.33, 0.33)
    
    
    for rf_w in np.linspace(0.3, 0.7, 5):  
        for cnn_w in np.linspace(0.2, 0.6, 5):
            lgb_w = 1.0 - cnn_w - rf_w
            if lgb_w < 0.05:  
                continue
                
            # Weighted prediction
            ensemble_preds = cnn_w*cnn_preds + lgb_w*lgb_preds + rf_w*rf_preds
            
            threshold = find_optimal_threshold(y_val, ensemble_preds, min_recall=0.98)
            ensemble_binary = (ensemble_preds >= threshold).astype(int)
            
            prec = precision_score(y_val, ensemble_binary)
            rec = recall_score(y_val, ensemble_binary)
            f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0
            
            if f2 > best_score:
                best_score = f2
                best_weights = (cnn_w, lgb_w, rf_w)
    
    print(f"Optimal ensemble weights: CNN={best_weights[0]:.2f}, LGB={best_weights[1]:.2f}, RF={best_weights[2]:.2f}")
    return best_weights

def train_models(X_specs, X_feats, y_labels, file_paths):
    """Train CNN, LightGBM, and Random Forest models with hyperparameter optimization"""
    X_train_specs, X_test_specs, X_train_feats, X_test_feats, y_train, y_test,file_paths_train, file_paths_test = train_test_split(
        X_specs, X_feats, y_labels, file_paths,test_size=0.2, random_state=SEED, stratify=y_labels
    )
    

    scaler = StandardScaler()
    X_train_feats_scaled = scaler.fit_transform(X_train_feats)
    X_test_feats_scaled = scaler.transform(X_test_feats)
    
    def objective_cnn(trial):
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
        
        # Build model with trial hyperparameters
        model = build_simple_cnn(X_train_specs.shape[1:])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_recall', patience=10, mode='max',restore_best_weights=True
        )
        
        history = model.fit(
            X_train_specs, y_train,
            epochs=50, 
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            callbacks=[early_stopping,MetricChecker()],
            verbose=0  
        )
        
        return max(history.history['val_accuracy'])
    
    def objective_lgb(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'verbosity': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        
        #  5-fold cross-validation for more robust evaluation
        score = cross_val_score(
            model, X_train_feats_scaled, y_train, 
            cv=5, 
            scoring='neg_log_loss',
            n_jobs=-1
        ).mean()
        
        return score
    
    def objective_rf(trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 300)  
        max_depth = trial.suggest_int('max_depth', 10, 30)  
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10) 
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)  
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        
        feature_type = trial.suggest_categorical('feature_type', ['categorical', 'float'])
        if feature_type == 'categorical':
            max_features = trial.suggest_categorical('max_features_categorical', ['sqrt', 'log2', None])
        else:
            max_features = trial.suggest_float('max_features_float', 0.3, 1.0)  
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=SEED,
            n_jobs=-1,
            class_weight='balanced'  
        )
        
        divided_feats = []
        for feat in X_train_feats:
            mid_point = len(feat) // 2
            divided = np.array([feat[:mid_point], feat[mid_point:]])
            divided_feats.append(divided.flatten())
        
        divided_feats = np.array(divided_feats)

        def expected_cost_scorer(y_true, y_pred, **kwargs):
            
            FN_COST = 15  # Cost of missing a fall (much higher)
            FP_COST = 1   # Cost of false alarm
            # Count false negatives (predicted 0, actual 1)
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            # Count false positives (predicted 1, actual 0)
            fp = np.sum((y_pred == 1) & (y_true == 0))
            
            #  total cost
            total_cost = (fn * FN_COST) + (fp * FP_COST)
            return -total_cost  # Negative because sklearn maximizes scores

        custom_scorer = make_scorer(expected_cost_scorer)
        
        # 5-fold CV for more robust evaluation
        score = cross_val_score(
            model, divided_feats, y_train, 
            cv=5,  
            scoring=custom_scorer,
            n_jobs=1
        ).mean()
        
        return score

    try:
        print("\nOptimizing CNN hyperparameters with Optuna...")
        study_cnn = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )
        study_cnn.optimize(objective_cnn, n_trials=25)  # Increased trials
        
        print(f"Best CNN parameters: {study_cnn.best_params}")
        print(f"Best CNN accuracy: {study_cnn.best_value:.4f}")
        
        best_cnn_params = study_cnn.best_params
        cnn_model = build_simple_cnn(X_train_specs.shape[1:])
        cnn_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=best_cnn_params.get('learning_rate', 0.0001)),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
        )
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {0: 1.0, 1: 5.0}  
        print(f"Using custom class weights: {class_weight_dict}")
        X_train_specs_calib, X_val_specs, y_train_calib, y_val = train_test_split(
            X_train_specs, y_train, test_size=0.15, stratify=y_train, random_state=SEED
        )

        calibrated_cnn = calibrate_model_with_temperature(cnn_model, X_val_specs, y_val)
        history = cnn_model.fit(
            X_train_specs, y_train,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            class_weight=class_weight_dict,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ],
            epochs=30,
            verbose=1
        )
    
    except Exception as e:
        print(f"Error in CNN optimization: {e}")
        cnn_model = build_simple_cnn(X_train_specs.shape[1:])
        history = cnn_model.fit(
            X_train_specs, y_train,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            epochs=10,
            verbose=1
        )
    
    try:
        print("\nOptimizing LightGBM hyperparameters with Optuna...")
        study_lgb = optuna.create_study(
            direction='maximize',  
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )
        study_lgb.optimize(objective_lgb, n_trials=25)  # Increased trials
        
        print(f"Best LightGBM parameters: {study_lgb.best_params}")
        print(f"Best LightGBM negative logloss: {study_lgb.best_value:.4f}")
        
        best_lgb_params = study_lgb.best_params
        lgb_model = lgb.LGBMClassifier(**best_lgb_params, class_weight={0: 1.0, 1: 3.0}, importance_type='gain')
        lgb_model.fit(X_train_feats_scaled, y_train)
    except Exception as e:
        print(f"Error in LightGBM optimization: {e}")
        lgb_model = lgb.LGBMClassifier()
        lgb_model.fit(X_train_feats_scaled, y_train)
    
    try:
        print("\nOptimizing Random Forest hyperparameters with Optuna...")
        study_rf = optuna.create_study(
            direction='maximize',  # We're using negative cost, so maximize
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )
        study_rf.optimize(objective_rf, n_trials=25) 
        
        print(f"Best Random Forest parameters: {study_rf.best_params}")
        print(f"Best Random Forest negative cost: {study_rf.best_value:.4f}")
        
        best_rf_params = study_rf.best_params
        
        if best_rf_params.get('feature_type') == 'categorical':
            max_features = best_rf_params.get('max_features_categorical')
        else:
            max_features = best_rf_params.get('max_features_float')
        
        rf_model = RandomForestClassifier(
            n_estimators=best_rf_params.get('n_estimators', 100),
            max_depth=best_rf_params.get('max_depth'),
            min_samples_split=best_rf_params.get('min_samples_split', 2),
            min_samples_leaf=best_rf_params.get('min_samples_leaf', 1),
            max_features=max_features,
            bootstrap=best_rf_params.get('bootstrap', True),
            random_state=SEED,
            n_jobs=-1,
            class_weight={0: 1.0, 1: 5.0}   # Added class weight
        )
    except Exception as e:
        print(f"Error in Random Forest optimization: {e}")
        rf_model = RandomForestClassifier(random_state=SEED, class_weight='balanced')
    
    # Process divided features for RF
    divided_feats_train = []
    for feat in X_train_feats:
        mid_point = len(feat) // 2
        divided = np.array([feat[:mid_point], feat[mid_point:]])
        divided_feats_train.append(divided.flatten())
    
    divided_feats_train = np.array(divided_feats_train)
    
    rf_model.fit(divided_feats_train, y_train)
    
    divided_feats_test = []
    for feat in X_test_feats:
        mid_point = len(feat) // 2
        divided = np.array([feat[:mid_point], feat[mid_point:]])
        divided_feats_test.append(divided.flatten())
    
    divided_feats_test = np.array(divided_feats_test)
    
    # Make predictions with each model
    cnn_preds = expit(calibrated_cnn.predict(X_test_specs).flatten())

    lgb_preds = lgb_model.predict_proba(X_test_feats_scaled)[:, 1]
    rf_preds = rf_model.predict_proba(divided_feats_test)[:, 1]
    

    ensemble_weights = optimize_ensemble_weights(
        {
            'cnn_model': cnn_model, 
            'lgb_model': lgb_model, 
            'rf_model': rf_model
        },
        [X_train_specs, X_train_feats_scaled, divided_feats_train],  
        y_train,
        metric='f2'  
    )
    ensemble_preds = (ensemble_weights[0] * cnn_preds + 
                  ensemble_weights[1] * lgb_preds + 
                  ensemble_weights[2] * rf_preds)
    
    precision, recall, thresholds = precision_recall_curve(y_test, ensemble_preds)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = find_optimal_threshold(y_test, ensemble_preds, min_recall=0.95)
    print(f"Optimal threshold (prioritizing recall): {optimal_threshold:.4f}")

    global ensemble_binary
    ensemble_binary = (ensemble_preds >= optimal_threshold).astype(int)
    
    # Evaluate ensemble
    print("\nEnsemble Model Performance:")
    print(classification_report(y_test, ensemble_binary))
    print(f"Ensemble AUC: {roc_auc_score(y_test, ensemble_preds):.4f}")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print("✅ Finished evaluating ensemble. Proceeding to visualizations...")

    try:
        os.makedirs('results/optimization', exist_ok=True)

        plot_model_visualizations(study_cnn, "cnn")
        plot_model_visualizations(study_lgb, "lgb")
        plot_model_visualizations(study_rf, "rf")

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, ensemble_preds)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, ensemble_preds):.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig('results/roc_curve.png')
        plt.close()

        # Confusion Matrix
        cm = confusion_matrix(y_test, ensemble_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Fall'], yticklabels=['Normal', 'Fall'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'results/confusion_matrix_{timestamp}.png')
        plt.close()

        # Calibration Curve
        plt.figure(figsize=(10, 8))
        prob_true, prob_pred = calibration_curve(y_test, cnn_preds, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=f'CNN (Brier: {brier_score_loss(y_test, cnn_preds):.3f})')

        prob_true, prob_pred = calibration_curve(y_test, lgb_preds, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='s', label=f'LightGBM (Brier: {brier_score_loss(y_test, lgb_preds):.3f})')

        prob_true, prob_pred = calibration_curve(y_test, rf_preds, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='^', label=f'Random Forest (Brier: {brier_score_loss(y_test, rf_preds):.3f})')

        prob_true, prob_pred = calibration_curve(y_test, ensemble_preds, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='*', linewidth=2,
                 label=f'Ensemble (Brier: {brier_score_loss(y_test, ensemble_preds):.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plots')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.savefig('results/calibration_plot.png')
        plt.close()

        # Extra results visualization (conf matrix, ROC, PR, etc.)
        visualize_results(y_test, ensemble_binary, ensemble_preds, history)
        print("Analyzing missed fall cases...")
        analyze_missed_cases(
            X_test_specs, X_test_feats, y_test, ensemble_binary, 
            file_paths_test, 
            {
                'cnn_model': cnn_model,
                'lgb_model': lgb_model,
                'rf_model': rf_model
            },
            output_dir='results/missed_cases'
        )
    except Exception as e:
        print(f"❌ Error in visualization: {e}")
        import traceback
        traceback.print_exc()
    return {
        'cnn_model': cnn_model,
        'lgb_model': lgb_model,
        'rf_model': rf_model,
        'scaler': scaler,
        'threshold': optimal_threshold
    }

def train_with_cross_validation(X_specs, X_feats, y_labels, n_folds=5):
    from sklearn.model_selection import StratifiedKFold
    
    cv_metrics = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': [],
        'log_loss': [], 'brier_score': []
    }
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Lists to store models for potential ensemble
    cnn_models = []
    lgb_models = []
    thresholds = []
    
    # Store calibration data across folds
    all_y_test = []
    all_ensemble_preds = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_specs, y_labels)):
        print(f"\n{'='*20} Fold {fold+1}/{n_folds} {'='*20}")
        
        X_train_specs, X_test_specs = X_specs[train_idx], X_specs[test_idx]
        X_train_feats, X_test_feats = X_feats[train_idx], X_feats[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        print(f"Class weights: {class_weight_dict}")
        
        cnn_model = build_simple_cnn(X_train_specs.shape[1:])
        
        cnn_model.fit(
            X_train_specs, y_train,
            epochs=100,  
            batch_size=8,  
            validation_split=0.2,
            class_weight=class_weight_dict,  
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ],
            verbose=1
        )
        
        scaler = StandardScaler()
        X_train_feats_scaled = scaler.fit_transform(X_train_feats)
        X_test_feats_scaled = scaler.transform(X_test_feats)
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',  
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train_feats_scaled, y_train)
        
        # Make predictions
        cnn_preds = expit(cnn_model.predict(X_test_specs).flatten())
        lgb_preds = lgb_model.predict_proba(X_test_feats_scaled)[:, 1]
        
        ensemble_preds = (cnn_preds + lgb_preds) / 2
        
        all_y_test.extend(y_test)
        all_ensemble_preds.extend(ensemble_preds)
        
        precision, recall, pr_thresholds = precision_recall_curve(y_test, ensemble_preds)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = pr_thresholds[optimal_idx] if len(pr_thresholds) > optimal_idx else 0.5

        
        ensemble_binary = (ensemble_preds >= optimal_threshold).astype(int)
        
        acc = accuracy_score(y_test, ensemble_binary)
        prec = precision_score(y_test, ensemble_binary)
        rec = recall_score(y_test, ensemble_binary)
        f1 = f1_score(y_test, ensemble_binary)
        auc = roc_auc_score(y_test, ensemble_preds)
        logloss = log_loss(y_test, ensemble_preds)
        brier = brier_score_loss(y_test, ensemble_preds)
        
        cv_metrics['accuracy'].append(acc)
        cv_metrics['precision'].append(prec)
        cv_metrics['recall'].append(rec)
        cv_metrics['f1'].append(f1)
        cv_metrics['auc'].append(auc)
        cv_metrics['log_loss'].append(logloss)
        cv_metrics['brier_score'].append(brier)
        
        cnn_models.append(cnn_model)
        lgb_models.append(lgb_model)
        thresholds.append(optimal_threshold)
        
        # Print fold results
        print(f"Fold {fold+1} Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Log Loss: {logloss:.4f}")
        print(f"  Brier Score: {brier:.4f}")
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
    
    # Print average metrics
    print("\nCross-Validation Results:")
    for metric, values in cv_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"  {metric.capitalize()}: {mean:.4f} ± {std:.4f}")
    
    # Plot cross-validation metrics
    try:
        plt.figure(figsize=(12, 8))
        for i, (metric, values) in enumerate(cv_metrics.items()):
            plt.subplot(3, 3, i+1)
            plt.bar(range(1, n_folds+1), values)
            plt.axhline(y=np.mean(values), color='r', linestyle='-')
            plt.title(f'{metric.capitalize()}')
            plt.xlabel('Fold')
            plt.ylabel('Score')
            plt.ylim(0, 1 if metric != 'log_loss' else None)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/cross_validation_metrics.png')
        plt.close()
        
        # Plot calibration curve for cross-validation
        plt.figure(figsize=(8, 6))
        prob_true, prob_pred = calibration_curve(np.array(all_y_test), np.array(all_ensemble_preds), n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot (Cross-Validation)')
        plt.grid(True, alpha=0.3)
        plt.savefig('results/cv_calibration_plot.png')
        plt.close()
    except Exception as e:
        print(f"Error plotting cross-validation metrics: {e}")
    
    return {
        'metrics': cv_metrics,
        'models': {
            'cnn': cnn_models,
            'lgb': lgb_models
        },
        'thresholds': thresholds
    }

def evaluate_model_stability(X_specs, X_feats, y_labels, n_runs=5):
    """Evaluate model stability across multiple runs with different random seeds"""
    # Metrics to track
    stability_metrics = {
        'accuracy': [], 'precision': [], 'recall': [], 
        'f1': [], 'auc': [], 'log_loss': [], 'brier_score': []
    }
    
    X_train_specs, X_test_specs, X_train_feats, X_test_feats, y_train, y_test = train_test_split(
        X_specs, X_feats, y_labels, test_size=0.2, random_state=SEED, stratify=y_labels
    )
    
    print(f"\nRunning stability analysis with {n_runs} runs...")
    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}")
        
        run_seed = SEED + run
        np.random.seed(run_seed)
        tf.random.set_seed(run_seed)
        
        cnn_model = build_simple_cnn(X_train_specs.shape[1:])
        cnn_model.fit(
            X_train_specs, y_train,
            epochs=30,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ],
            verbose=0
        )
        
        scaler = StandardScaler()
        X_train_feats_scaled = scaler.fit_transform(X_train_feats)
        X_test_feats_scaled = scaler.transform(X_test_feats)
        
        # Train LightGBM
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            random_state=run_seed
        )
        lgb_model.fit(X_train_feats_scaled, y_train)
        
        # Make predictions
        cnn_preds = expit(cnn_model.predict(X_test_specs).flatten())
        lgb_preds = lgb_model.predict_proba(X_test_feats_scaled)[:, 1]
        
        # Ensemble predictions
        ensemble_preds = (cnn_preds + lgb_preds) / 2
        ensemble_binary = (ensemble_preds >= 0.5).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(y_test, ensemble_binary)
        prec = precision_score(y_test, ensemble_binary)
        rec = recall_score(y_test, ensemble_binary)
        f1 = f1_score(y_test, ensemble_binary)
        auc = roc_auc_score(y_test, ensemble_preds)
        logloss = log_loss(y_test, ensemble_preds)
        brier = brier_score_loss(y_test, ensemble_preds)
        
        # Store metrics
        stability_metrics['accuracy'].append(acc)
        stability_metrics['precision'].append(prec)
        stability_metrics['recall'].append(rec)
        stability_metrics['f1'].append(f1)
        stability_metrics['auc'].append(auc)
        stability_metrics['log_loss'].append(logloss)
        stability_metrics['brier_score'].append(brier)
        
        # Print run results
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
    
    # Calculate stability statistics
    for metric, values in stability_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        cv = std / mean if mean > 0 else 0  # Coefficient of variation
        print(f"\n{metric.capitalize()} Stability:")
        print(f"  Mean: {mean:.4f}")
        print(f"  Std Dev: {std:.4f}")
        print(f"  CV: {cv:.4f}")
    
    # Plot stability metrics
    try:
        plt.figure(figsize=(12, 8))
        for i, (metric, values) in enumerate(stability_metrics.items()):
            if i >= 6:  # Only plot first 6 metrics
                break
            plt.subplot(2, 3, i+1)
            plt.plot(range(1, n_runs+1), values, marker='o', label=metric.capitalize())
            plt.axhline(y=np.mean(values), color='r', linestyle='-')
            plt.title(f'{metric.capitalize()} Stability')
            plt.xlabel('Run')
            plt.ylabel('Score')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/model_stability.png')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        for metric, values in stability_metrics.items():
            plt.plot(range(1, n_runs+1), values, marker='o', label=metric.capitalize())
        
        plt.xlabel('Run')
        plt.ylabel('Score')
        plt.title('Model Stability Across Multiple Runs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/model_stability_combined.png')
        plt.close()
    except Exception as e:
        print(f"Error plotting stability metrics: {e}")
    
    return stability_metrics

def model_averaging(X_specs, X_feats, y_labels, n_models=5):
    """Train multiple models and average their predictions"""
    # Split data
    X_train_specs, X_test_specs, X_train_feats, X_test_feats, y_train, y_test = train_test_split(
        X_specs, X_feats, y_labels, test_size=0.2, random_state=SEED, stratify=y_labels
    )
    
    scaler = StandardScaler()
    X_train_feats_scaled = scaler.fit_transform(X_train_feats)
    X_test_feats_scaled = scaler.transform(X_test_feats)
    
    cnn_models = []
    lgb_models = []
    scalers = []
    all_preds = []
    
    individual_f1 = []
    individual_auc = []
    individual_logloss = []
    individual_brier = []
    individual_cost = []
    
    for i in range(n_models):
        print(f"\nTraining Model {i+1}/{n_models}")
        
        seed = SEED + i
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Create a new scaler for each model
        scaler_i = StandardScaler()
        X_train_feats_scaled_i = scaler_i.fit_transform(X_train_feats)
        X_test_feats_scaled_i = scaler_i.transform(X_test_feats)
        
        # Train CNN
        cnn_model = build_simple_cnn(X_train_specs.shape[1:])
        cnn_model.fit(
            X_train_specs, y_train,
            epochs=50,
            batch_size=8,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ],
            verbose=0
        )
        
        # Train LightGBM
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            random_state=seed
        )
        lgb_model.fit(X_train_feats_scaled_i, y_train)
        
        # Make predictions
        cnn_preds = expit(cnn_model.predict(X_test_specs).flatten())

        lgb_preds = lgb_model.predict_proba(X_test_feats_scaled_i)[:, 1]
        
        # Ensemble predictions for this model
        model_preds = (cnn_preds + lgb_preds) / 2
        model_binary = (model_preds >= 0.5).astype(int)
        
        # Calculate metrics for this model
        f1 = f1_score(y_test, model_binary)
        auc = roc_auc_score(y_test, model_preds)
        logloss = log_loss(y_test, model_preds)
        brier = brier_score_loss(y_test, model_preds)
        
        # Calculate expected cost
        fn = np.sum((model_binary == 0) & (y_test == 1))
        fp = np.sum((model_binary == 1) & (y_test == 0))
        cost = (fn * 10) + (fp * 1)  # FN costs 10x more than FP
        
        # Store metrics
        individual_f1.append(f1)
        individual_auc.append(auc)
        individual_logloss.append(logloss)
        individual_brier.append(brier)
        individual_cost.append(cost)
        
        # Store models and predictions
        cnn_models.append(cnn_model)
        lgb_models.append(lgb_model)
        scalers.append(scaler_i)
        all_preds.append(model_preds)
        
        # Print model results
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Log Loss: {logloss:.4f}")
        print(f"  Expected Cost: {cost:.1f}")
    
    # Average predictions
    all_preds = np.array(all_preds)
    averaged_preds = np.mean(all_preds, axis=0)
    
    precision, recall, thresholds = precision_recall_curve(y_test, averaged_preds)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5
    
    averaged_binary = (averaged_preds >= optimal_threshold).astype(int)
    
    # Calculate expected cost for averaged model
    fn = np.sum((averaged_binary == 0) & (y_test == 1))
    fp = np.sum((averaged_binary == 1) & (y_test == 0))
    expected_cost = (fn * 10) + (fp * 1)
    
    # Print comparison
    print("\nModel Averaging Results:")
    print(f"Average of Individual F1: {np.mean(individual_f1):.4f} ± {np.std(individual_f1):.4f}")
    print(f"Averaged Model F1: {f1_score(y_test, averaged_binary):.4f}")
    print(f"Average of Individual AUC: {np.mean(individual_auc):.4f} ± {np.std(individual_auc):.4f}")
    print(f"Averaged Model AUC: {roc_auc_score(y_test, averaged_preds):.4f}")
    print(f"Average of Individual Log Loss: {np.mean(individual_logloss):.4f} ± {np.std(individual_logloss):.4f}")
    print(f"Averaged Model Log Loss: {log_loss(y_test, averaged_preds):.4f}")
    print(f"Average of Individual Cost: {np.mean(individual_cost):.1f} ± {np.std(individual_cost):.1f}")
    print(f"Averaged Model Cost: {expected_cost}")
    
    # Plot comparison
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(range(n_models), individual_f1, alpha=0.6, label='Individual Models')
        plt.axhline(y=f1_score(y_test, averaged_binary), color='r', linestyle='-', label='Averaged Model')
        plt.xlabel('Model Index')
        plt.ylabel('F1 Score')
        plt.title('Model Averaging Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/model_averaging.png', dpi=300)
        plt.close()
        
        # Plot calibration curve for model averaging
        plt.figure(figsize=(8, 6))
        
        # Individual models calibration (average)
        all_prob_true = []
        all_prob_pred = []
        for preds in all_preds:
            prob_true, prob_pred = calibration_curve(y_test, preds, n_bins=10)
            all_prob_true.append(prob_true)
            all_prob_pred.append(prob_pred)
        
        avg_prob_true = np.mean(all_prob_true, axis=0)
        avg_prob_pred = np.mean(all_prob_pred, axis=0)
        
        plt.plot(avg_prob_pred, avg_prob_true, marker='o', linestyle='--', 
                 label=f'Avg Individual (Brier: {np.mean(individual_brier):.3f})')
        
        # Averaged model calibration
        prob_true, prob_pred = calibration_curve(y_test, averaged_preds, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='s', linewidth=2, 
                 label=f'Averaged Model (Brier: {brier_score_loss(y_test, averaged_preds):.3f})')
        
        # Add diagonal perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot - Model Averaging')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.savefig('results/model_averaging_calibration.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error plotting model averaging comparison: {e}")
    
    return {
        'cnn_models': cnn_models,
        'lgb_models': lgb_models,
        'scalers': scalers,
        'threshold': optimal_threshold,
        'individual_performance': {
            'f1': individual_f1, 
            'auc': individual_auc,
            'log_loss': individual_logloss,
            'brier_score': individual_brier,
            'expected_cost': individual_cost
        },
        'averaged_performance': {
            'f1': f1_score(y_test, averaged_binary),
            'auc': roc_auc_score(y_test, averaged_preds),
            'log_loss': log_loss(y_test, averaged_preds),
            'brier_score': brier_score_loss(y_test, averaged_preds),
            'expected_cost': expected_cost
        }
    }

def visualize_mel_spectrograms(audio_file, output_dir='results'):
    """Generates visual spectrograms for research/presentation"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        y, sr = librosa.load(audio_file, sr=SR)
        
        # Generate standard mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, n_fft=2048, 
            hop_length=512, fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Generate with different colormaps
        plt.figure(figsize=(10, 6))
        for i, cmap in enumerate(['viridis', 'inferno', 'plasma', 'magma']):
            plt.subplot(2, 2, i+1)
            librosa.display.specshow(
                mel_spec_db, y_axis='mel', x_axis='time', 
                sr=sr, fmax=8000, cmap=cmap
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel Spectrogram ({cmap})')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/mel_spectrograms_{timestamp}.png', dpi=300)
        plt.close()
        
        for cmap in ['viridis', 'inferno']:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                mel_spec_db, y_axis='mel', x_axis='time', 
                sr=sr, fmax=8000, cmap=cmap
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel Spectrogram')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/spectrogram_{cmap}_{timestamp}.png', dpi=300)
            plt.close()
            
        return f'{output_dir}/mel_spectrograms_{timestamp}.png'
    except Exception as e:
        print(f"Error visualizing mel spectrograms: {e}")
        return None

def compare_normal_fall_spectrograms(normal_file, fall_file, output_dir='results'):
    """Creates side-by-side comparison of normal vs fall spectrograms"""
    try:
        y1, sr1 = librosa.load(normal_file, sr=SR)
        y2, sr2 = librosa.load(fall_file, sr=SR)
        
        spec1 = librosa.feature.melspectrogram(y=y1, sr=sr1, n_mels=N_MELS, fmax=8000)
        spec2 = librosa.feature.melspectrogram(y=y2, sr=sr2, n_mels=N_MELS, fmax=8000)
        
        spec1_db = librosa.power_to_db(spec1, ref=np.max)
        spec2_db = librosa.power_to_db(spec2, ref=np.max)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        vmin = min(spec1_db.min(), spec2_db.min())
        vmax = max(spec1_db.max(), spec2_db.max())
        
        img1 = librosa.display.specshow(spec1_db, x_axis='time', y_axis='mel', 
                                sr=sr1, fmax=8000, ax=axes[0], cmap='viridis',
                                vmin=vmin, vmax=vmax)
        axes[0].set_title('Normal Audio')
        
        librosa.display.specshow(spec2_db, x_axis='time', y_axis='mel', 
                                sr=sr2, fmax=8000, ax=axes[1], cmap='viridis',
                                vmin=vmin, vmax=vmax)
        axes[1].set_title('Fall Audio')
        
        fig.colorbar(img1, ax=axes, format='%+2.0f dB')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.tight_layout()
        plt.savefig(f'{output_dir}/normal_vs_fall_{timestamp}.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error comparing spectrograms: {e}")

def plot_model_visualizations(study, model_name):
    """Plot optimization visualizations for a model and save as HTML instead of PNG (no kaleido needed)."""
    try:
        import optuna.visualization as vis
        os.makedirs('results/optimization', exist_ok=True)

        try:
            fig = vis.plot_optimization_history(study)
            fig.write_html(f'results/optimization/{model_name}_optimization_history.html')
            print(f"Saved: {model_name}_optimization_history.html")
        except Exception as e:
            print(f"[{model_name}] Failed to plot optimization history: {e}")

        try:
            fig = vis.plot_param_importances(study)
            fig.write_html(f'results/optimization/{model_name}_param_importances.html')
            print(f"Saved: {model_name}_param_importances.html")
        except Exception as e:
            print(f"[{model_name}] Failed to plot param importances: {e}")

        try:
            param_importances = optuna.importance.get_param_importances(study)
            top_params = list(param_importances.keys())[:3]
            if top_params:
                fig = vis.plot_slice(study, params=top_params)
                fig.write_html(f'results/optimization/{model_name}_slice_plot.html')
                print(f"Saved: {model_name}_slice_plot.html")
        except Exception as e:
            print(f"[{model_name}] Failed to plot slice: {e}")

        # Contour plot
        try:
            if len(top_params) >= 2:
                fig = vis.plot_contour(study, params=top_params[:2])
                fig.write_html(f'results/optimization/{model_name}_contour_plot.html')
                print(f"Saved: {model_name}_contour_plot.html")
        except Exception as e:
            print(f"[{model_name}] Failed to plot contour: {e}")

    except Exception as e:
        print(f"❌ Error generating {model_name} visualizations: {e}")

def visualize_results(y_true, y_pred, y_probs, history=None, audio_files=None):
    """Create comprehensive visualizations of model results"""
    try:
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Part 1: Model Performance Visualizations
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Fall'], yticklabels=['Normal', 'Fall'])
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        plt.savefig(f'results/confusion_matrix_{timestamp}.png', dpi=600)
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, 
                 label=f'ROC Curve (AUC = {roc_auc_score(y_true, y_probs):.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'results/roc_curve_{timestamp}.png', dpi=600)
        plt.close()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2,
                label=f'PR Curve (AP = {np.mean(precision):.4f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'results/precision_recall_curve_{timestamp}.png', dpi=600)
        plt.close()
        
        # Calibration Curve
        plt.figure(figsize=(8, 6))
        prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2,
                label=f'Calibration Curve (Brier Score: {brier_score_loss(y_true, y_probs):.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Plot', fontsize=14)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'results/calibration_curve_{timestamp}.png', dpi=600)
        plt.close()
        
        # Histogram of Prediction Probabilities
        plt.figure(figsize=(10, 6))
        plt.hist(y_probs[y_true==0], bins=20, alpha=0.5, color='blue', label='Normal')
        plt.hist(y_probs[y_true==1], bins=20, alpha=0.5, color='red', label='Fall')
        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Distribution of Prediction Probabilities', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'results/prediction_distribution_{timestamp}.png', dpi=600)
        plt.close()
        
        # Part 2: Training History Visualization (if available)
        if history is not None and hasattr(history, 'history'):
            # Training & Validation Loss
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Training and Validation Loss', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.savefig(f'results/training_loss_{timestamp}.png', dpi=600)
            plt.close()
            
            # Training & Validation Accuracy
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in history.history:
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title('Training and Validation Accuracy', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.savefig(f'results/training_accuracy_{timestamp}.png', dpi=600)
            plt.close()
            
            # Training & Validation Precision/Recall (if available)
            if 'precision' in history.history and 'recall' in history.history:
                plt.figure(figsize=(10, 6))
                plt.plot(history.history['precision'], label='Training Precision')
                plt.plot(history.history['recall'], label='Training Recall')
                if 'val_precision' in history.history and 'val_recall' in history.history:
                    plt.plot(history.history['val_precision'], label='Validation Precision')
                    plt.plot(history.history['val_recall'], label='Validation Recall')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Score', fontsize=12)
                plt.title('Training and Validation Precision/Recall', fontsize=14)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.savefig(f'results/training_precision_recall_{timestamp}.png', dpi=600)
                plt.close()
        
        # Part 3: Feature Importance Visualization (if available)
        # This would typically come from the LightGBM or Random Forest model
        
        print(f"Visualizations saved to results/ directory with timestamp {timestamp}")
    except Exception as e:
        print(f"Error in visualize_results: {e}")
        import traceback
        traceback.print_exc()
def analyze_missed_cases(X_specs, X_feats, y_true, y_pred, file_paths, models, output_dir='results/missed_cases'):
    """Analyze cases where falls were missed (false negatives)"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find indices of false negatives (missed falls)
    fn_indices = np.where((y_pred == 0) & (y_true == 1))[0]
    
    if len(fn_indices) == 0:
        print("No missed falls to analyze!")
        return
    
    print(f"Analyzing {len(fn_indices)} missed falls...")
    
    # Collect statistics about missed falls
    missed_stats = {
        'avg_amplitude': [],
        'peak_frequency': [],
        'spectral_centroid': [],
        'zero_crossing_rate': [],
        'file_names': []
    }
    
    # Process each missed fall
    for i, idx in enumerate(fn_indices):
        spec = X_specs[idx]
        feat = X_feats[idx]
        file_path = file_paths[idx]
        file_name = os.path.basename(file_path)
        
        # Load audio for analysis
        try:
            audio, sr = load_audio_file(file_path, target_sr=SR)
            if audio is not None:
                avg_amp = np.mean(np.abs(audio))
                stft = np.abs(librosa.stft(audio))
                peak_freq_bin = np.argmax(np.mean(stft, axis=1))
                peak_freq = librosa.fft_frequencies(sr=sr, n_fft=2048)[peak_freq_bin]
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
                zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
                
                missed_stats['avg_amplitude'].append(avg_amp)
                missed_stats['peak_frequency'].append(peak_freq)
                missed_stats['spectral_centroid'].append(spectral_centroid)
                missed_stats['zero_crossing_rate'].append(zcr)
                missed_stats['file_names'].append(file_name)
                
                # Generate and save spectrogram
                plt.figure(figsize=(10, 6))
                S_db = librosa.power_to_db(np.abs(librosa.stft(audio)), ref=np.max)
                librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Missed Fall: {file_name}')
                plt.savefig(os.path.join(output_dir, f'missed_fall_{i}_{file_name}.png'))
                plt.close()
                
        except Exception as e:
            print(f"Error analyzing missed fall {file_name}: {e}")
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.hist(missed_stats['avg_amplitude'], bins=10)
    plt.title('Average Amplitude')
    
    plt.subplot(2, 2, 2)
    plt.hist(missed_stats['peak_frequency'], bins=10)
    plt.title('Peak Frequency (Hz)')
    
    plt.subplot(2, 2, 3)
    plt.hist(missed_stats['spectral_centroid'], bins=10)
    plt.title('Spectral Centroid')
    
    plt.subplot(2, 2, 4)
    plt.hist(missed_stats['zero_crossing_rate'], bins=10)
    plt.title('Zero Crossing Rate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missed_falls_statistics.png'))
    plt.close()
    
    summary_df = pd.DataFrame(missed_stats)
    summary_df.to_csv(os.path.join(output_dir, 'missed_falls_summary.csv'), index=False)
    
    print(f"Analysis of {len(fn_indices)} missed falls completed.")
    print(f"Average amplitude: {np.mean(missed_stats['avg_amplitude']):.4f}")
    print(f"Average spectral centroid: {np.mean(missed_stats['spectral_centroid']):.4f}")
    print(f"Results saved to {output_dir}")
def generate_calibration_report(y_true, model_predictions, model_names):
    """Generate comprehensive calibration report for model predictions"""
    brier_scores = []
    log_losses = []
    
    plt.figure(figsize=(12, 8))
    
    for i, (preds, name) in enumerate(zip(model_predictions, model_names)):
        prob_true, prob_pred = calibration_curve(y_true, preds, n_bins=10)
        brier = brier_score_loss(y_true, preds)
        logloss = log_loss(y_true, preds)
        
        brier_scores.append(brier)
        log_losses.append(logloss)
        
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, 
                 label=f'{name} (Brier: {brier:.4f}, LogLoss: {logloss:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curves Comparison', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('results/calibration_report.png', dpi=300)
    plt.close()
    
    print("\nCalibration Report:")
    for name, brier, ll in zip(model_names, brier_scores, log_losses):
        print(f"{name}: Brier Score={brier:.4f}, Log Loss={ll:.4f}")

    #  metrics table
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Brier Score': brier_scores,
        'Log Loss': log_losses
    })
    return metrics_df

def main():
    try:
        import optuna
    except ImportError:
        print("Installing optuna...")
        import subprocess
        subprocess.check_call(["pip", "install", "optuna", "plotly", "kaleido"])
        import optuna
    print("Enhanced Fall Detection System with Stability Analysis")
    print("====================================================")
    
    print("\nStep 0: Clearing feature cache to regenerate with new extraction method...")
    cache_dir = os.path.join(os.path.dirname(DATA_DIR), 'cache')
    shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"Cache cleared: {not os.path.exists(cache_dir)}")
    
    print("\nStep 1: Preparing dataset...")
    X_specs, X_feats, y_labels, file_paths = prepare_dataset()
    
    if len(X_specs) == 0:
        print("ERROR: Dataset preparation failed.")
        return
    
    # 1. Cross-validation for stable performance estimates
    print("\nStep 2: Performing cross-validation...")
    cv_results = train_with_cross_validation(X_specs, X_feats, y_labels, n_folds=5)
    
    # 2. Model stability across multiple runs
    print("\nStep 3: Analyzing model stability...")
    stability_results = evaluate_model_stability(X_specs, X_feats, y_labels, n_runs=5)
    # 3. Model averaging
    print("\nStep 4: Model averaging...")
    averaging_results = model_averaging(X_specs, X_feats, y_labels, n_models=5)
    
    # Step 5: Train final production model
    print("\nStep 5: Training final production model...")
    X_train_specs, X_test_specs, X_train_feats, X_test_feats, y_train, y_test = train_test_split(
        X_specs, X_feats, y_labels, test_size=0.3, random_state=SEED, stratify=y_labels
    )
    print(f"Spectrogram shape: {X_specs.shape}")
    final_model = train_models(X_specs, X_feats, y_labels, file_paths)
    print("🎉 Final model training complete!")
    print("Returned model keys:", list(final_model.keys()))

    # === Step 6: Extensive testing block ===
    if RUN_EXTENSIVE_TEST:
        print("\nStep 6: Extensive testing...")
        test_results = []
        correct_count = 0
        total_count = 0

        # sample of files for testing
        normal_files = [f for f in os.listdir(NORMAL_DIR) if f.endswith(('.mp3', '.wav'))][:20]
        abnormal_files = [f for f in os.listdir(ABNORMAL_DIR) if f.endswith(('.mp3', '.wav'))][:20]

        # Test normal files
        print("\nTesting normal files...")
        for file in normal_files:
            try:
                file_path = os.path.join(NORMAL_DIR, file)
                audio, sr = load_audio_file(file_path, target_sr=SR)
                if audio is None:
                    print(f"Skipping {file} - loading failed")
                    continue
                
                # Extract features
                spec = create_spectrogram(audio=audio, sr=sr)
                feat = extract_features(audio, sr)
                
                if spec is None or feat is None:
                    print(f"Skipping {file} - feature extraction failed")
                    continue
                
                spec = np.expand_dims(spec, axis=0)
                feat = np.expand_dims(feat, axis=0)
                
                feat_scaled = final_model['scaler'].transform(feat)
                
                # Make predictions
                cnn_pred = final_model['cnn_model'].predict(spec).flatten()[0]
                lgb_pred = final_model['lgb_model'].predict_proba(feat_scaled)[0, 1]
                
                # Process divided features for RF
                mid_point = len(feat[0]) // 2
                divided = np.array([feat[0][:mid_point], feat[0][mid_point:]])
                divided = divided.flatten().reshape(1, -1)
                rf_pred = final_model['rf_model'].predict_proba(divided)[0, 1]
                
                # Ensemble prediction
                ensemble_pred = (0.4 * cnn_pred) + (0.3 * lgb_pred) + (0.3 * rf_pred)
                binary_pred = 1 if ensemble_pred >= final_model['threshold'] else 0
                
                # Check correctness (0 = normal)
                is_correct = binary_pred == 0
                if is_correct:
                    correct_count += 1
                total_count += 1
                
                # Store result
                test_results.append({
                    'file': file,
                    'true_label': 'normal',
                    'prediction': 'fall' if binary_pred == 1 else 'normal',
                    'probability': ensemble_pred,
                    'correct': is_correct
                })
                
                print(f"File: {file}, Pred: {binary_pred}, Prob: {ensemble_pred:.4f}, Correct: {is_correct}")
                generate_calibration_report(
                            y_test,
                            [cnn_pred, lgb_pred, rf_pred, ensemble_pred],
                            ['CNN', 'LightGBM', 'Random Forest', 'Ensemble'])
                # Visualize some examples
                if len(test_results) <= 5:
                    visualize_mel_spectrograms(file_path, output_dir='results/test_examples')
            except Exception as e:
                print(f"Error testing {file}: {e}")
        
        # Test abnormal files
        print("\nTesting abnormal files...")
        for file in abnormal_files:
            try:
                file_path = os.path.join(ABNORMAL_DIR, file)
                audio, sr = load_audio_file(file_path, target_sr=SR)
                if audio is None:
                    print(f"Skipping {file} - loading failed")
                    continue
                
                # Extract features
                spec = create_spectrogram(audio=audio, sr=sr)
                feat = extract_features(audio, sr)
                
                if spec is None or feat is None:
                    print(f"Skipping {file} - feature extraction failed")
                    continue
                
                # Reshape for model input
                spec = np.expand_dims(spec, axis=0)
                feat = np.expand_dims(feat, axis=0)
                
                # Scale features
                feat_scaled = final_model['scaler'].transform(feat)
                
                # Make predictions
                cnn_pred = expit(final_model['cnn_model'].predict(spec).flatten()[0])
                lgb_pred = final_model['lgb_model'].predict_proba(feat_scaled)[0, 1]
                
                # Process divided features for RF
                mid_point = len(feat[0]) // 2
                divided = np.array([feat[0][:mid_point], feat[0][mid_point:]])
                divided = divided.flatten().reshape(1, -1)
                rf_pred = final_model['rf_model'].predict_proba(divided)[0, 1]
                
                # Ensemble prediction
                ensemble_pred = (0.4 * cnn_pred) + (0.3 * lgb_pred) + (0.3 * rf_pred)
                binary_pred = 1 if ensemble_pred >= final_model['threshold'] else 0
                
                # Check correctness (1 = abnormal/fall)
                is_correct = binary_pred == 1
                if is_correct:
                    correct_count += 1
                total_count += 1
                
                # Store result
                test_results.append({
                    'file': file,
                    'true_label': 'fall',
                    'prediction': 'fall' if binary_pred == 1 else 'normal',
                    'probability': ensemble_pred,
                    'correct': is_correct
                })
                
                print(f"File: {file}, Pred: {binary_pred}, Prob: {ensemble_pred:.4f}, Correct: {is_correct}")
                
                # Visualize some examples
                if len(test_results) <= 10:
                    visualize_mel_spectrograms(file_path, output_dir='results/test_examples')
            except Exception as e:
                print(f"Error testing {file}: {e}")
        
        # Calculate overall accuracy
        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"\nTest Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
        analyze_missed_cases(X_test_specs, X_test_feats, y_test, binary_pred, file_paths, final_model)
        # Save test results to CSV
        try:
            import csv
            with open('results/test_results.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['file', 'true_label', 'prediction', 'probability', 'correct'])
                writer.writeheader()
                writer.writerows(test_results)
            print("Test results saved to results/test_results.csv")
        except Exception as e:
            print(f"Error saving test results: {e}")
        
        # Compare normal and fall spectrograms
        try:
            if normal_files and abnormal_files:
                normal_file = os.path.join(NORMAL_DIR, normal_files[0])
                fall_file = os.path.join(ABNORMAL_DIR, abnormal_files[0])
                compare_normal_fall_spectrograms(normal_file, fall_file)
                print("Spectrogram comparison saved to results/")
        except Exception as e:
            print(f"Error comparing spectrograms: {e}")
    else:
        print("\n[Skipped Step 6: Extensive testing]")
    
    print("\nStep 7: Saving final model...")
    # Save models
    try:
        os.makedirs('models', exist_ok=True)
        final_model['cnn_model'].save('models/cnn_model.keras')

        
        # Save LightGBM model
        import joblib
        joblib.dump(final_model['lgb_model'], 'models/lgb_model.joblib')
        joblib.dump(final_model['rf_model'], 'models/rf_model.joblib')
        joblib.dump(final_model['scaler'], 'models/scaler.joblib')
        
        # Save threshold
        with open('models/threshold.txt', 'w') as f:
            f.write(str(final_model['threshold']))
        
        print("\nFall Detection System training complete!")
        print(f"Final model saved to 'models/' directory")
        print(f"Test results saved to 'results/test_results.csv'")
        print(f"Visualizations saved to 'results/' directory")
    except Exception as e:
        print(f"Error saving models: {e}")

if __name__ == "__main__":
    main()