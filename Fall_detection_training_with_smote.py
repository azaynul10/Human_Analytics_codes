# Add impact detection features
def extract_impact_features(audio_array, sr):
    onset_env = librosa.onset.onset_strength(y=audio_array, sr=sr)
    return [np.max(onset_env), np.mean(onset_env)]
# Update this to your actual dataset path
DATADIR = 'c:\Users\Abedi\OneDrive - Student Ambassadors\archive (7)\combined_datasets'  # Changed from fan dataset
CATEGORIES = ['fall', 'non_fall']  # More descriptive for your use case
def create_training_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for audio in os.listdir(path):
            audio_path = os.path.join(path, audio)
            audio_array, sr = librosa.load(audio_path, duration=30, sr=22050)  # Explicit SR
            
            # Add fall-specific features
            impact_features = extract_impact_features(audio_array, sr)
            
            # Existing features
            time_features = extract_time_domain_features(audio_array)
            freq_features = extract_frequency_domain_features(audio_array)
            
            features = time_features + freq_features + impact_features
            training_data.append([features, class_num])
    return training_data
# Add after train_test_split
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

