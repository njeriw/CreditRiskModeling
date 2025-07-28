# Loan Status Prediction and Credit Risk Assessment

## Introduction
This project focuses on developing a robust machine learning model to predict loan status and assess credit risk using a comprehensive dataset of Lending Club loan applications. It highlights advanced data preprocessing techniques, with a particular emphasis on leveraging modern natural language processing (NLP) methods for textual features.

The core objective is to build a predictive model that can classify loan applications into different statuses (e.g., "Fully Paid", "Charged Off", "Current") at the time of application. This involves transforming raw, heterogeneous loan data into a suitable format for machine learning algorithms and then training and evaluating a classification model.

The project utilizes a dataset of Lending Club loan applications. This dataset contains a mix of numerical, categorical, and textual features describing applicant demographics, financial history, loan characteristics, and the ultimate loan status.

## Key Features & Components

__Comprehensive Data Preprocessing:__

Word2Vec: Utilized for generating static word embeddings, capturing semantic relationships based on word co-occurrence.

BERT (Bidirectional Encoder Representations from Transformers): Employed for creating rich, contextualized word embeddings, enabling a deeper understanding of textual nuances in features like emp_title (if applicable).

High Cardinality Feature Management: Strategies to mitigate overfitting risks associated with features like emp_title, including consideration of text processing and embeddings.

Scikit-learn Pipelines & ColumnTransformer: Efficiently orchestrating all preprocessing steps within a unified pipeline to ensure consistency and prevent data leakage.

Missing Value Imputation: Strategic handling of missing data, including KNN Imputation for numerical features and specialized approaches for highly sparse joint applicant data (e.g., creating a binary indicator).

Outlier Analysis: Identification of outliers in numerical features, recognizing their potential as crucial risk indicators.

Zero/Low Variance Feature Handling: Identification and removal of uninformative columns. 

Categorical Encoding: Application of OrdinalEncoder for ordered categories (grade, sub_grade) and OneHotEncoder for nominal features.

Numerical Scaling: Standardization of numerical features using RobustScaler.

__Model Training & Interpretation:__

Random Forest Classifier: Ensemble method model chosen for its inherent capability to provide feature importance scores for feature selection.

Feature Importance Analysis: Extraction and visualization of feature importances to identify the most influential factors in loan status prediction. 

__Model Evaluation:__

Rigorous evaluation of the model's performance using appropriate classification metrics (e.g., Accuracy, Precision, Recall, F1-score, ROC-AUC) to assess its predictive capability and generalization.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Results & Insights 📈 
Data Structure: The dataset comprises approximately 10,000 samples and 55 features, with a mix of numerical and object (categorical/text) data types.

Target Imbalance: The loan_status target variable exhibited a severe class imbalance (mode is "Current"), which will require careful handling during model training (e.g., sampling techniques, appropriate loss functions).

High Cardinality: Features like emp_title were identified as high cardinality, necessitating advanced text processing.

Textual Insights: The application of Word2Vec and BERT provided richer representations for textual data, enhancing the model's ability to capture nuanced information from fields like emp_title.

Missing Data: Significant missingness was observed in joint applicant features. Missing values handled with binary indicators rather than imputation.

Zero Variance: Columns such as num_accounts_120d_past_due were found to have zero variance, indicating they are uninformative and were dropped.

Outliers: Several key risk indicators (delinq_2y, num_historical_failed_to_pay, total_collection_amount_ever, public_record_bankrupt) showed high outlier percentages, which are considered valuable predictive signals.

Multivariate Analysis: Revealed strong correlations between loan_amount and installment, and surprisingly, a trend of higher loan amounts for lower credit grades (A-G). Loan purpose also significantly influenced loan amount distribution.


## Future Work ⏭️
Hyperparameter Tuning: Implement more extensive hyperparameter tuning for the Random Forest Classifier and potentially other models.

Class Imbalance Handling: Implement specific strategies to address the class imbalance in the target variable (e.g., SMOTE, undersampling).

Model Interpretability: Deep dive into SHAP or LIME for more granular model explanations beyond global feature importance.

Model Monitoring: Monitor of model performance every six months to detect data drift.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Built with 
This project is built using `Python 3.12` and relies on several key libraries for its functionality, including `pandas` and `numpy` for efficient data manipulation and numerical operations, and `scikit-learn` for various machine learning utilities such as data splitting and performance evaluation. 
All necessary dependencies can be installed via pip using the command: `pip install torch pandas numpy scikit-learn sklearn`.
<p align="left">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=sklearn,anaconda" />
  </a>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



