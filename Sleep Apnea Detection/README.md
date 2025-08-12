# Sleep Apnea Detection

Model that detects Sleep Apnea using HRV (Heart Rate Variability) features extracted from ECG (Electrocardiogram) signals.
>Sleep apnea is a medical condition where the person suffers from intermittent breathing cessations during sleep. It's correlated with increased mortality rate, myocardial infarctions, and increased blood pressure.
Project was made as part of my medical engineering and health tech studies.

1. Loads ECG signals, filters them using Butterworth, detects signals' R-peaks
2. Extracts HRV features
3. Trains and tunes RandomForest and XGBClassifier with GridSearch
4. Evaluates with Accuracy, F1 and ROC-AUC

#### Results

| Model         | Accuracy | F1 (Apnea) | ROC-AUC |
|---------------|----------|------------|---------|
| Dummy (majority) | 0.52 | 0.00 | 0.50 |
| RandomForest  | **0.82** | **0.82** | **0.88** |
| XGBoost       | 0.782 | 0.78 | 0.87 |
