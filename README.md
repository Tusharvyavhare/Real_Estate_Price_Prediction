# 🏠 Real Estate Price Prediction

## 📌 Project Overview

This project predicts real estate property prices using Machine Learning techniques. The model analyzes various property features such as location, area, number of bedrooms, bathrooms, parking availability, lift facility, security, garden, and other amenities to estimate the property's market value.

The objective is to assist buyers, sellers, and real estate professionals in making data-driven decisions.

---

## 🚀 Features

- Property price prediction using Machine Learning
- Data preprocessing and cleaning
- Feature engineering
- Model training and evaluation
- Interactive prediction system
- Support for multiple property amenities
- Visualization of data and model performance

---

## 📊 Dataset Information

The dataset contains property-related information including:

- Property Area (sq.ft)
- Number of Bedrooms
- Number of Bathrooms
- Parking Availability
- Lift Facility
- Security
- Garden
- Location
- Property Type
- Target Variable: Property Price

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib

### Development Environment
- Jupyter Notebook
- VS Code

---

## 📂 Project Structure

```
Real-Estate-Price-Prediction/
│
├── data/
│   └── merged_real_estate_data.csv
│
├── notebooks/
│   └── model_training.ipynb
│
├── models/
│   └── price_prediction_model.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
│
├── images/
│   └── project_output.png
│
├── requirements.txt
├── README.md
└── app.py
```

---

## ⚙️ Machine Learning Workflow

### 1. Data Collection
Collected real estate property data from reliable sources.

### 2. Data Cleaning
- Removed missing values
- Handled duplicate records
- Corrected inconsistent data

### 3. Feature Engineering
- Encoded categorical variables
- Scaled numerical features
- Selected important features

### 4. Model Training
The following algorithms were tested:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

### 5. Model Evaluation

Evaluation Metrics:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

## 📈 Results

| Metric | Value |
|----------|----------|
| R² Score | XX.XX |
| MAE | XX.XX |
| RMSE | XX.XX |

*Replace the values above with your actual results.*

---

## 📷 Project Screenshots

### Data Analysis
(Add screenshot here)

### Model Performance
(Add screenshot here)

### Prediction Interface
(Add screenshot here)

---

## 🔮 Future Improvements

- Integration with live real estate data
- Web-based prediction dashboard
- Advanced feature engineering
- Deep Learning models
- Location-based price prediction using maps

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Real-Estate-Price-Prediction.git
```

### Navigate to Project Folder

```bash
cd Real-Estate-Price-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Project

```bash
python app.py
```

---

## 💡 Example Prediction

Input:

- Area: 1200 sq.ft
- Bedrooms: 2
- Bathrooms: 2
- Parking: Yes
- Lift: Yes
- Security: Yes
- Garden: No

Predicted Price:

₹ 65,00,000

---

## 👨‍💻 Author

**Tushar Vyavahare**

Computer Engineering Student  
Machine Learning & Java Enthusiast

GitHub: https://github.com/your-username

---

## 📜 License

This project is licensed under the MIT License.
