# Student Stress Monitoring

Model that predicts students' stress types using academic, personal, social and environmental factors from the dataset. 
>Distress is considered "bad stress" that the individual may find overwhelming or frightening.
>Eustress is considered "good stress" that the individual may find challenging and exciting.

Dataset was provided by Kaggle and the project was done as an exercise to improve data analysis and ML skills.

1. Loads the data, checks for NaN values and outliers and cleans them
2. Encodes the target variable Stress Types
3. Splits the data and oversamples target variable's class minorities
4. Trains and tunes RandomForest and XGBClassifier with GridSearch
5. Evaluates with Confusion Matrices and Macro F1 and compares the scores to baseline models

#### Results

| Model         | Macro F1 | 
|---------------|----------|
| Dummy | 0.23 |
| RandomForest  | 0.66 |
| XGBoost       | **0.89** |
