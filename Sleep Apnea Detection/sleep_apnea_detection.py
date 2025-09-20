# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Koulu
#     language: python
#     name: koulu
# ---

# # Sleep Apnea Detection

# This project was part of the Machine Learning Health Technology Project course in University of Turku. 
#
# Data was given along with the assignment and it includes a single channel ECG signal measured during sleep. 
#
# - Sampling rate is 100hz and samples are 30 seconds long.
# - Objective of the project was to create a model that can detect sleep apnea using HRV features extracted from ECG data. 
# > Sleep apnea is a medical condition where the person suffers from intermittent breathing cessations during sleep. It's correlated with increased mortality rate, myocardial infarctions, and increased blood pressure.
#

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import neurokit2 as nk
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from scipy.stats import randint, uniform
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# ## Data loading

# We begin by first globbing and reading the data from text files. We do this by writing a simple read_txt function that we use to iterate through the lists.
#
# We begin by reading the ECG signals from .txt files using a custom read_txt function and glob.
# Normal and Apnea signals are separated in their own directories.

# +
datadir = Path("data")

apnea_train = list((datadir / "Train/apnea").glob("*.txt"))
normal_train = list((datadir / "Train/normal").glob("*.txt"))
apnea_test = list((datadir / "Test/apnea").glob("*.txt"))
normal_test = list((datadir / "Test/normal").glob("*.txt"))

def read_txt(data):
    signals = []
    for i in data:
        df = pd.read_csv(i, header=None)
        signals.append(df.squeeze())
    return signals

apnea_train = read_txt(apnea_train)
normal_train = read_txt(normal_train)
apnea_test = read_txt(apnea_test)
normal_test = read_txt(normal_test)

# -

# ## Filtering & Peak detection

# We apply a Butterworth band-pass filter (0.5-40 Hz) to remove baseline wandering and high frequency noise. We're also utilizing neurokit2's ecg_peaks to locate R-peaks in the filtered signals.

# +
def butter_bandpass(signal, lowcut=0.5, highcut=40.0, fs=100, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

def find_peaks(filtered_signal):
    r_peaks_list = []
    for signal in filtered_signal:
        _, rpeaks = nk.ecg_peaks(signal, sampling_rate=100)
        r_peaks_list.append(rpeaks["ECG_R_Peaks"])
    return r_peaks_list

def plot_signals(signals, peaks_list, title, n=3):
    plt.figure(figsize=(12, 6))
    for i in range(min(n, len(signals))):
        plt.plot(signals[i][:1000], label=f"Sample ID {i+1}", alpha=0.8)
        p = [idx for idx in peaks_list[i] if idx < 1000]
        plt.scatter(p, signals[i][p], label=f"R-peaks sample ID {i+1}", alpha=0.8)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.show()

def preprocess(signals):
    filtered = [butter_bandpass(signal) for signal in signals]
    peaks = find_peaks(filtered)
    return filtered, peaks

normal_train_filtered, normal_train_peaks = preprocess(normal_train)
normal_test_filtered, normal_test_peaks = preprocess(normal_test)
apnea_train_filtered, apnea_train_peaks = preprocess(apnea_train)
apnea_test_filtered, apnea_test_peaks = preprocess(apnea_test)

plot_signals(apnea_train_filtered, apnea_train_peaks, "Training Samples, Apnea, Filtered (100hz)", n=3)
plot_signals(normal_train_filtered, normal_train_peaks, "Training Samples, Normal, Filtered (100hz)", n=3)
# -

# Here we check that the neurokit2's hrv method is correctly computing the HRV data for our peaks' data

display(nk.hrv(normal_train_peaks[0], sampling_rate=100))
display(nk.hrv(apnea_train_peaks[0], sampling_rate=100))
# ## Feature extraction
X_train_hrv = []
for peak in normal_train_peaks + apnea_train_peaks:
    HRV_data = nk.hrv(peak, sampling_rate=100)
    X_train_hrv.append(HRV_data.iloc[0])

X_test_hrv = []
for peak in normal_test_peaks + apnea_test_peaks:
    HRV_data = nk.hrv(peak, sampling_rate=100)
    X_test_hrv.append(HRV_data.iloc[0])

# Here we do the features and labels for our models' data. We label normal ECGs with 0 and apnea ECGs with 1. After creating the model data we standardize it.
X_train = pd.DataFrame(X_train_hrv)
y_train = np.array([0] * len(normal_train) + [1] * len(apnea_train))
X_test = pd.DataFrame(X_test_hrv)
y_test = np.array([0] * len(normal_test) + [1] * len(apnea_test))

X_train.info()
X_test.info()
# ['HRV_SDANN2', 'HRV_SDNNI2', 'HRV_SDANN5, 'HRV_SDNNI5', 'HRV_ULF'] are completely empty
X_train.dropna(axis=1, how='all', inplace=True)
X_test.dropna(axis=1, how='all', inplace=True)



# +

models = {
    'DummyClassifier' : {
        'estimator' : DummyClassifier(random_state=42),
        'params' : {
            'clf__strategy' : ['most_frequent', 'stratified', 'uniform']
        }
    },
    'RandomForest' : {
        'estimator' : RandomForestClassifier(random_state=42),
        'params' : {
            'clf__n_estimators': randint(100, 801),
            'clf__max_depth': [None] + list(np.arange(1, 31)),
            'clf__min_samples_split': randint(2, 8),
            'clf__min_samples_leaf': randint(1, 8),
            'clf__bootstrap': [True, False]
        }
    },
    'XGBoost' : {
        'estimator' : XGBClassifier(eval_metric='logloss', random_state=42),
        'params' : {
            'clf__n_estimators': randint(100, 801),
            'clf__max_depth': randint(1, 7),
            'clf__learning_rate': uniform(0.01, 0.3),
            'clf__subsample': uniform(0.6, 0.4),
            'clf__colsample_bytree': uniform(0.6, 0.4)
        } 
    }
}

inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

model_scores = {}

for name, cfg in models.items():
    print(f"\n --- \n Now tuning {name}\n --- \n")
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', cfg['estimator'])
    ])
    randomsearch = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=cfg['params'],
        cv=inner_cv,
        scoring='recall',
        n_iter=30,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    scores = cross_val_score(randomsearch, X_train, y_train, cv=outer_cv, scoring='recall', n_jobs=1)
    model_scores[name] = f"Mean {np.mean(scores)} +- {np.std(scores)} std"

model_scores
best_model = models['XGBoost']

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', best_model['estimator'])
])

randomsearch_final = RandomizedSearchCV(
        estimator = pipe,
        param_distributions = best_model['params'],
        cv = inner_cv,
        scoring = 'recall',
        n_iter = 100,
        n_jobs = -1,
        verbose = 2,
        random_state = 42
)

randomsearch_final.fit(X_train, y_train)
xgb = randomsearch_final.best_estimator_
print(randomsearch_final.best_params_)
print(randomsearch_final.best_score_) 


clf = randomsearch_final.best_estimator_.named_steps['clf']
importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

feature_importance_df.head(20)
features = np.array(feature_importance_df['feature'].head(20)) # we'll use these to rerun the model

X_train = X_train[features]
X_test = X_test[features]

X_train.info()
X_test.info()

randomsearch_final.fit(X_train, y_train)
xgb = randomsearch_final.best_estimator_
print(randomsearch_final.best_params_)
print(randomsearch_final.best_score_)

# ### Evaluation
# Performance was evaluated using Accuracy, Precision, Recall, F1-score and ROC+AUC

forest_classification = classification_report(y_val, y_pred_forest)
forest_confusion = confusion_matrix(y_val, y_pred_forest)
xgb_classification = classification_report(y_val, y_pred_xgb)
xgb_confusion = confusion_matrix(y_val, y_pred_xgb)

# Here is a plot for the Confusion Matrices.

# +
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(forest_confusion, display_labels=['Normal', 'Apnea']).plot(ax=axs[0], cmap='Blues')
axs[0].set_title('RandomForest Confusion Matrix')

ConfusionMatrixDisplay(xgb_confusion, display_labels=['Normal', 'Apnea']).plot(ax=axs[1], cmap='Reds')
axs[1].set_title('XGBoost Confusion Matrix')

plt.tight_layout()
plt.show()
# -

# Here are the classification reports' stats.

print(forest_classification)
print(xgb_classification)

# Finally, let's plot the ROC Curve for both models.

# +

rf_probs = forest.predict_proba(X_test_scaled)[:, 1]
xgb_probs = xgb.predict_proba(X_test_scaled)[:, 1]

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)

rf_auc = auc(rf_fpr, rf_tpr)
xgb_auc = auc(xgb_fpr, xgb_tpr)

plt.plot(rf_fpr, rf_tpr, label=f'Random Forest {rf_auc:.2f}')
plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost {xgb_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
# -

# # Conclusion
#
# | Model         | Accuracy | F1 (Apnea) | ROC-AUC |
# |---------------|----------|------------|---------|
# | Dummy (majority) | 0.52 | 0.00 | 0.50 |
# | RandomForest  | **0.82** | **0.82** | **0.88** |
# | XGBoost       | 0.782 | 0.78 | 0.87 |
#
# When detecting sleep apnea with this data set, we can see that RandomForest is performing better. Data got filtered using Butterworth band-pass and ECG got handled using neurokit2's methods. After adjusting the selected HRV features to be used with the models we found that involving neurokit2's HRV method's all features brought the best results.
