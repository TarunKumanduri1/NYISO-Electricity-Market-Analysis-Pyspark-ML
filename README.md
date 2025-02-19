# **NYISO Electricity Market Analysis**

## **Overview**
This project analyzes New York Independent System Operator (NYISO) electricity market data, focusing on electricity pricing, demand forecasting, congestion analysis, and anomaly detection. It includes machine learning and deep learning models implemented using PySpark and TensorFlow.

## **Dataset**
The dataset consists of two primary components:
- **Load Data**: Contains historical electricity demand (integrated load) for different pricing nodes.
- **Price Data**: Provides locational-based marginal prices (LBMP), congestion costs, and transmission losses.

### **Key Features**
1. **PTID (Pricing Node ID)** – Unique identifier for pricing locations.
2. **Load (MWh)** – Total electricity demand at a given time.
3. **LBMP ($/MWHr)** – Locational-based marginal price at a specific node.
4. **Marginal Cost of Losses** – Additional cost due to energy losses in transmission.
5. **Marginal Cost of Congestion** – Extra cost due to transmission constraints.



## **Project Workflow**
### **1. Data Preprocessing & Feature Engineering**
- Converted timestamps into structured date and time components.
- Applied log transformation, lag features, rolling averages, and interaction terms.
- Encoded time-of-day and weekend indicators.
- Scaled and standardized numerical features for ML models.

### **2. Data Storage & Processing with PySpark**
- Combined multiple CSV files into Parquet format for faster query execution.
- Used PySpark for large-scale data transformations and efficient distributed processing.

### **3. Predicting Electricity Prices using Gradient Boosted Trees**
- Used historical demand and price data to train a **Gradient Boosted Trees (GBT)** model.
- Handled outliers using the **Interquartile Range (IQR) Method**.
- Evaluated model performance using **RMSE and R²**.

### **4. Detecting Anomalies in Electricity Prices using Clustering**
- Applied **KMeans clustering** to detect unusual price spikes.
- Used rolling averages and standard deviations to determine deviations.
- Defined a threshold for anomalies based on the Euclidean distance to cluster centers.

### **5. Forecasting Future Electricity Demand using LSTM**
- Prepared sequential data using a **24-hour rolling window**.
- Normalized features using `StandardScaler`.
- Trained a **Long Short-Term Memory (LSTM) neural network** using TensorFlow/Keras.
- Used **Early Stopping** to prevent overfitting.



## **Results & Key Findings**
- **Electricity Price Prediction**: Achieved an **R² score of 0.9334**, indicating high model accuracy.
- **Anomaly Detection**: Successfully identified unusual price spikes due to congestion or sudden demand changes.
- **Demand Forecasting**: The **LSTM model achieved an R² of 0.9559**, demonstrating strong predictive capabilities.



## **Setup & Installation**
### **Prerequisites**
Ensure you have **Python 3.8+** installed along with `pip`.

### **Installation Steps**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repository/NYISO-Analysis.git
   cd NYISO-Analysis
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # For MacOS/Linux
   env\Scripts\activate  # For Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Spark application:**
   ```bash
   python analysis.py
   ```



## **Technologies Used**
- **PySpark** – Data processing and machine learning at scale.
- **TensorFlow/Keras** – Deep learning model for demand forecasting.
- **Scikit-learn** – Clustering and statistical modeling.
- **Matplotlib & Seaborn** – Data visualization.
- **Pandas & NumPy** – Data handling and transformation.



## **Project Structure**
```
NYISO-Analysis/
│── data/                  # Raw and processed datasets
│── notebooks/             # Jupyter notebooks for exploration
│── models/                # Trained ML models
│── src/
│   ├── data_processing.py # Data cleaning and feature engineering
│   ├── price_prediction.py # GBT Model for price forecasting
│   ├── anomaly_detection.py # KMeans clustering for anomalies
│   ├── demand_forecasting.py # LSTM model for load prediction
│── README.md
│── requirements.txt
│── analysis.py            # Main execution script
```



## **Authors & Acknowledgments**
Developed by **[Your Name]**  
Data sourced from **NYISO**  
Thanks to **Apache Spark & TensorFlow communities** for their open-source contributions.



## **License**
This project is licensed under the **MIT License**.


This `requirements.txt` includes:
- **PySpark** for distributed data processing.
- **TensorFlow/Keras** for deep learning models.
- **Scikit-learn** for clustering and anomaly detection.
- **Pandas & NumPy** for data handling.
- **Matplotlib & Seaborn** for visualization.
