# 📈 Market Sentiment Prediction from Trader Behavior

Predict and visualize **market sentiment (Fear/Greed)** using trader-level data from Hyperliquid and the Bitcoin Fear & Greed Index.

> 🚀 Built with Scikit-learn, Streamlit, Pandas, and real-world Web3 trading data.

---

## 📌 Project Objective

Explore how **trader performance** and behavior correlates with **market sentiment** and build an end-to-end system that:

- 🧼 Cleans and merges multi-source financial datasets
- 📊 Performs statistical testing & rolling window feature engineering
- 🤖 Trains a predictive model to forecast sentiment
- 🌐 Hosts a live dashboard for predictions and insights

---

## 🗂️ Project Structure
```
market-sentiment-trading-analysis/
├── data/                           # Raw and processed data files
│   ├── fear_greed_index.csv           # Raw sentiment data
│   ├── historical_data.csv            # Raw trader data
│   ├── merged_trader_sentiment_data.csv # Preprocessed & merged data
│   └── feature_engineered_data.csv    # Cleaned + feature-rich dataset
│
├── src/                            # Source code for data processing & modeling
│   ├── clean_and_merge.py             # Data cleaning & joining script
│   ├── statistical_tests.py           # T-tests & correlation analysis
│   ├── feature_engineering.py         # Rolling stats + new features
│   ├── predict_sentiment_model.py     # Model training & evaluation
│   └── improve_model.py               # Optimized model
│
├── outputs/                        # Generated outputs
│   ├── models/                         # Trained models and scalers
│   │   ├── sentiment_rf_model.pkl
│   │   ├── sentiment_rf_model_optimized.pkl
│   │   └── scaler.pkl
│   │
│   ├── eda/                            # Visualizations from exploratory analysis
│   │   ├── pnl_over_time.png
│   │   ├── sentiment_distribution.png
│   │   ├── trade_count_over_time.png
│   │   └── volume_over_time.png
│   │
│   └── statstics/                      # Statistical test results
│       ├── statistical_tests.csv
│       └── correlation_results.csv
│
├── app/                            # Dashboard app for visualization
│   └── dashboard.py                   # Streamlit dashboard
│
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation

```

---

## 🚀 How to Run This Project

### 1. Clone the repository

```bash
git clone https://github.com/yashdew03/market-sentiment-trading-analysis.git
cd market-sentiment-trading-analysis
```

---

### 2. Set up your environment
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### 3. Run the files
```bash
python src/clean_and_merge.py
python src/statistical_tests.py
python src/feature_engineering.py
python src/predict_sentiment_model.py
python src/improve_model.py
```

---

### 4. Launch the Dashboard
```bash
streamlit run app/dashboard.py
```


--- 

## 🧠 Key Features

📈 EDA Charts: Visualize how trader behavior changes with sentiment

🧪 Statistical Testing: Validate if sentiment significantly affects trading metrics

🛠️ Feature Engineering: Rolling mean, std, trade volume patterns

🤖 Random Forest Model: Predict sentiment from trader metrics

🌐 Streamlit UI: Upload your own trade data and get sentiment predictions

---

## 🧩 Tech Stack

| Purpose           | Tools Used               |
| ----------------- | ------------------------ |
| Data Analysis     | `pandas`, `numpy`        |
| Statistical Tests | `scipy`, `statsmodels`   |
| Machine Learning  | `scikit-learn`, `joblib` |
| Visualization     | `matplotlib`, `seaborn`  |
| Dashboard         | `streamlit`              |

---

## 🔬 Machine Learning Pipeline

- **Data Cleaning**: Handled nulls, converted timestamps, merged datasets.
- **Feature Engineering**: Rolling windows (3D mean/std), trade count aggregations.
- **Statistical Testing**: T-tests, correlation significance.
- **Modeling**: Random Forest with scaling + inference pipeline.
- **Evaluation**: Confusion matrix, F1-score, recall/precision.

---

## 📊 Sample Output

#### Model Evaluation:
Accuracy = 81%

F1-score (Greed class) = 0.89
 
Random Forest was selected over others for interpretability and robustness.

---

## 📩 Requirements

- **Python 3.8+**
- Libraries listed in `requirements.txt`

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yashdew3/market-sentiment-trading-analysis/issues) (if you have one) or open a new issue to discuss changes. Pull requests are also appreciated.

---

## 📬 Contact
- Built by **Yash Dewangan**
- Github: [YashDewangan](https://github.com/yashdew3)
- Email: [yashdew06@gmail.com](yashdew06@gmail.com)

Enjoy using the Predictive Model for the Market Sentiment in Indian Stock Market! 🚀