# 🔊 AudioFallGuard: Advanced Fall Detection System 🚨

<p align="center">
  <img src="https://img.shields.io/github/last-commit/azaynul10/Human_Analytics_codes" alt="GitHub last commit">
  <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/TensorFlow-2.15-orange" alt="TensorFlow">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>
  


> "When you fall down, AI picks up the sound!" 🔊👂

A state-of-the-art ensemble-based system for detecting falls through audio analysis. This repository contains the code for an audio-based fall detection system achieving **97.9% accuracy** and **95.8% recall** - because missing a fall is worse than a false alarm!

## 📊 Performance at a Glance

```
Accuracy: 0.9792 ± 0.0000
Precision: 1.0000 ± 0.0000  
Recall: 0.9583 ± 0.0000
F1 Score: 0.9787 ± 0.0000
AUC: 0.9990 ± 0.0014
```

## 🧠 Model Architecture: Triple Threat Ensemble


  


This system combines three powerful models:

1. **3D CNN**: Processes spectrograms with multi-branch architecture and attention mechanism 👁️
2. **LightGBM**: Analyzes 286 handcrafted audio features for speed and accuracy 🚀
3. **Random Forest**: Handles divided feature vectors with custom cost function that penalizes missed falls 15× more than false alarms ⚖️

```python
# The magic of ensemble prediction
ensemble_preds = (0.4 * cnn_preds) + (0.3 * lgb_preds) + (0.3 * rf_preds)
```

## 🎯 Key Features

- **Dual-Input Modality**: Processes both spectrograms AND extracted features
- **Bayesian Hyperparameter Optimization**: Using Optuna for all three models
- **Fall-Prioritized Evaluation**: Custom cost functions that prioritize recall
- **Temperature Scaling**: Calibrates probability outputs for reliability
- **Extensive Testing Framework**: 5-fold CV, stability analysis, and more

## 🔍 Hyperparameter Optimization Journey


  


After running Optuna for days (and nights), we found these optimal hyperparameters:

```python
# Random Forest Best Params
{
    'n_estimators': 181, 
    'max_depth': 21, 
    'min_samples_split': 10,
    'min_samples_leaf': 3, 
    'bootstrap': False, 
    'max_features': 'log2'
}

# CNN Best Params
{
    'learning_rate': 0.0001, 
    'dropout_rate': 0.4, 
    'l2_reg': 0.001
}
```

## ⚡ Quick Start

```bash
# Clone repository
git clone https://github.com/azaynul10/Human_Analytics_codes.git
cd AudioFallGuard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the main system
python audio_fall_detector_cnn_lgbm_rf.py
```

## 📈 Model Performance Visualizations


  


## 💻 Code Snippets

```python
# Creating the Attention Mechanism
attention = tf.keras.layers.Conv3D(64, (1, 1, 1), activation='sigmoid')(merged)
x = tf.keras.layers.Multiply()([merged, attention])
```

## 🧪 Experiments I've Run

When you've spent months running experiments 24/7, you learn a few things:

- ✅ **Multi-branch CNN > Single-branch CNN**
- ✅ **Calibrated ensemble > Individual models**
- ✅ **F2-score threshold tuning > F1-score tuning**
- ❌ **LSTM by itself 
  


- [ ] Implement LSTM layers to capture temporal patterns
- [ ] Expand dataset with more diverse fall scenarios
- [ ] Deploy to edge devices with TensorFlow Lite
- [ ] Create real-time audio streaming interface
- [ ] Add integration with alert systems

## 💪 About Me


  


> Fueled by coffee and determination, I've spent countless sleepless nights optimizing these models!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---


  



  If this repository helped you, please consider giving it a ⭐!
