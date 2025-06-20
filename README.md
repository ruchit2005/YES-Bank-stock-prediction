# 📈 Yes Bank Stock Closing Price Prediction

> **Project Type:** Regression  
> **Author:** Ruchit Gupta  
> **Status:** Completed ✅  
> **GitHub Repository:** [Link to Project Repo](https://github.com/ruchit2005/YES-Bank-stock-prediction.git)

---

## 🧠 Overview

This project aims to **predict the monthly closing stock price of Yes Bank**, one of India’s leading financial institutions. The analysis is motivated by a major fraud case in 2018 involving the former CEO, Rana Kapoor, which drastically impacted the bank's financial standing and public trust.

Using machine learning models, this project forecasts future stock closing prices based on historical trends using attributes such as `Open`, `High`, `Low`, `Month`, and `Year`.

---

## 📊 Dataset

- **Source:** Custom CSV file with Yes Bank's historical stock data
- **Features:**
  - `Open`, `High`, `Low`, `Close` prices
  - `Date` (converted to `Month` and `Year`)
- **Target:** `Close` (monthly closing stock price)

---

## 🛠️ Technologies & Libraries

- Python 3
- Pandas & NumPy
- Scikit-learn
- Matplotlib
- Seaborn (optional)
- Pickle (for model saving)

---


---

## 🤖 Models Implemented

Four machine learning regression models were tested and compared:

| Model                  | Description                                              |
|------------------------|----------------------------------------------------------|
| 📐 Linear Regression    | Basic baseline linear model                              |
| 🧮 Ridge Regression     | Linear model with L2 regularization                      |
| 🌳 Random Forest        | Ensemble of decision trees                               |
| 📏 SVR (Support Vector) | Best-fitting margin with linear kernel                   |

---

## 📈 Evaluation Metrics

Models were evaluated using:

- **R² Score**: Goodness of fit
- **MAE (Mean Absolute Error)**: Average of absolute errors
- **MSE (Mean Squared Error)**: Penalizes larger errors

---

## 📉 Results Summary

- All models had **similar R² scores (~0.98)**.
- **SVR outperformed all others** in both MAE and MSE.
- **Random Forest** had the highest error—likely overfitting on the small dataset.

| Model           | R² Score | MAE ↓   | MSE ↓   |
|------------------|----------|--------|--------|
| Linear Regression| ~0.98    | High   | High   |
| Ridge Regression | ~0.98    | Moderate| Moderate|
| Random Forest    | ~0.98    | Highest| Highest|
| **SVR** (Best)   | ~0.98    | **Lowest** | **Lowest** |

---

## 🔬 Hypothesis Testing

A **paired t-test** was conducted between the absolute errors of **SVR** and **Linear Regression**:

- **Null Hypothesis (H₀)**: No difference in mean errors
- **Alternative Hypothesis (H₁)**: SVR performs significantly better
- **Result**: *p-value < 0.05* → SVR significantly outperforms Linear Regression ✅

---

## 💾 Model Deployment

- The **best model (SVR)** is saved using `pickle`:
  ```python
  import pickle
  pickle.dump(model4, open('best_model.pkl', 'wb'))


## 📌 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/ruchit2005/YES-Bank-stock-prediction.git
   cd YES-Bank-stock-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:

   ```bash
   jupyter notebook yes_bank_stock_prediction.ipynb
   ```

---

## 📚 Key Learnings

* Regression models are effective in modeling stock trends with clean and structured features.
* Support Vector Regression can outperform tree-based methods on smaller tabular datasets.
* Proper feature engineering (like splitting `Date` into `Month` and `Year`) is crucial for time-series-like tabular data.
* Visual comparisons and hypothesis testing help validate model performance beyond metrics.

---

## 📎 Future Improvements

* Use **Time-Series models** like ARIMA or LSTM for temporal modeling
* Add external features: market indicators, news sentiment, or volume
* Deploy using **Streamlit** or **Flask** for web-based prediction

---

## 🙌 Acknowledgments

* Thanks to [Scikit-learn documentation](https://scikit-learn.org/) for model references.
* Guided by personal curiosity and a passion for financial analytics and ML.

---

## 📫 Contact

**Ruchit Gupta**
📧 [Email](mailto:ruchit2005@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/ruchit2005/)
🐙 [GitHub](https://github.com/ruchit2005)

---

⭐ If you found this project helpful, please **star** the repo and share it!







