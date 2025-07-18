# 👩‍💼 Employee Attrition Prediction Using Machine Learning

A machine learning project focused on predicting employee attrition, identifying key influencing factors, and enabling HR teams to take proactive action using real-world IBM HR data.

## 📁 Files Included

- `HR-Employee-Attrition.csv` – Original IBM HR dataset  
- `Employee_Attrition (1).ipynb` – Full notebook with preprocessing, modeling, and evaluation  
- `Employee_Attrition (FINAL).pdf` – Research paper summarizing results and methodology

## 🧠 Techniques Used

### ✅ Supervised Learning Models
- **Gaussian Naive Bayes** *(Best Recall: 0.67)*
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost, Gradient Boosting
- K-Nearest Neighbors (KNN)

### 🧹 Preprocessing & Feature Engineering
- Categorical encoding (Label + One-Hot)
- Z-score normalization
- Class imbalance handled with **SMOTE**

## 📊 Visualizations (EDA)

- Attrition rates by age, gender, salary, department
- Correlation heatmaps for numerical features
- Outlier detection & treatment
- Insights on overtime and distance-from-home patterns

## 🚀 Environment & Tools

Developed in **Python** using:
- `pandas`, `numpy`, `scikit-learn`, `xgboost`
- `matplotlib`, `seaborn`, `plotly`
- `imbalanced-learn` (SMOTE)

## 📈 Results

| Model             | Accuracy | Recall (🔍 Priority) | Precision | F1 Score |
|------------------|----------|----------------------|-----------|----------|
| Naive Bayes       | 84.72%   | **0.67**             | 0.36      | 0.47     |
| Gradient Boosting | 89.01%   | 0.41                 | 0.64      | 0.50     |
| Random Forest     | 88.79%   | 0.41                 | 0.61      | 0.49     |
| SVM               | 90.14%   | 0.38                 | 0.76      | 0.50     |

📌 **Note**: Recall was prioritized to reduce false negatives and help HR identify at-risk employees effectively.

## 🧾 Citation

> Nihitha Sanikommu, S Dhanya Ratna Madhuri, Navya Kethavath, Kommineni Chandhana  
> *Predicting Employee Attrition Using Machine Learning Techniques*  
> VIT-AP University, 2025

---

📫 **For inquiries or collaboration:**  
[LinkedIn](https://www.linkedin.com/in/your-profile) | [Email](mailto:ratnamadhuri.22bce8698@vitapstudent.ac.in)
