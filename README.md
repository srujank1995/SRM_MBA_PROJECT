
# A STUDY ON AI-BASED PRODUCT DEMAND PREDICTION FOR E-COMMERCE PLATFORMS

## MBA Final Year Project – Artificial Intelligence and Data Science

**Institution**: SRM Institute of Science & Technology  
**Degree**: Master of Business Administration (MBA)  
**Specialization**: Artificial Intelligence and Data Science  
**Academic Year**: 2024-2026 | Semester 4  
**Date**: May 2026  

---

## 📋 Project Overview

This project presents a comprehensive machine learning solution for predicting product demand in e-commerce platforms. The system leverages advanced AI techniques, multiple predictive models, and business intelligence features to provide actionable insights for inventory management, revenue optimization, and strategic decision-making.

### Problem Statement
E-commerce businesses face significant challenges in:
- **Inventory Management**: Balancing stock levels to avoid stockouts and overstock situations
- **Demand Forecasting**: Accurately predicting future product demand based on historical patterns
- **Financial Planning**: Optimizing pricing strategies and profit margins
- **Supply Chain Optimization**: Coordinating with suppliers based on accurate demand predictions
- **Resource Allocation**: Efficiently allocating capital and warehouse space

This project addresses these challenges using machine learning models and business intelligence frameworks.

---

## 🎯 Project Objectives

1. **Develop Multiple Machine Learning Models** for demand prediction
2. **Compare Model Performance** across different algorithms
3. **Implement Advanced Feature Engineering** for improved predictions
4. **Create Intelligent Recommendation Engine** for business actions
5. **Design Real-time Alert System** for critical business situations
6. **Generate Financial Impact Analysis** for data-driven decisions
7. **Build Interactive Dashboard** for stakeholder visualization and decision-making

---

## ✨ Key Features

### 1. **Multiple Machine Learning Models**
- ✅ Random Forest Regressor
- ✅ XGBoost (Extreme Gradient Boosting)
- ✅ Gradient Boosting Regressor
- ✅ Ridge Regression (L2 Regularization)
- ✅ Linear Regression (Baseline)
- ✅ Neural Network (MLPRegressor)

### 2. **Advanced Data Processing**
- Data cleaning and preprocessing
- Missing value handling
- Categorical variable encoding (LabelEncoder)
- Feature scaling and normalization (StandardScaler)
- Outlier detection and removal
- Stratified sampling for large datasets

### 3. **Model Evaluation & Comparison**
- Cross-validation (5-fold CV)
- Multiple metrics: R² Score, MAE, RMSE, MAPE
- Train-test split (80-20)
- Performance ranking and comparison
- Model-specific confidence scores

### 4. **Intelligent Recommendations Engine** 🆕
- Demand classification (VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH)
- Context-aware action recommendations
- Priority-labeled suggestions (HIGH, MEDIUM, LOW)
- Financial impact analysis per recommendation
- Inventory optimization guidance

### 5. **Critical Alerts System** 🆕
- Real-time notifications for critical situations
- 4 severity levels: CRITICAL, WARNING, INFO, SUCCESS
- Alerts for: demand spikes, stockouts, profitability issues, opportunities
- Actionable recommendations with each alert
- Automatic threshold-based triggering

### 6. **Financial Impact Dashboard** 🆕
- Revenue forecasting and variance analysis
- Profit estimation and margin calculation
- ROI (Return on Investment) metrics
- Break-even analysis
- Inventory holding cost calculation
- Safety stock and reorder point recommendations
- Stock turnover rate analysis

### 7. **Interactive Streamlit Dashboard**
- 6 comprehensive tabs for different analyses
- Real-time predictions
- Model comparison visualizations
- Detailed performance metrics
- Financial forecasting tools
- Business conclusions and insights
- Professional UI with custom styling

---

## 📊 Project Structure

```
ai_demand_prediction_project/
│
├── 📄 app.py                          # Main Streamlit application (6 tabs)
├── 📄 train_model.py                  # Model training and evaluation
├── 📄 business_conclusions.py          # Business analyzer & recommendations (NEW)
│
├── 📊 data_new.csv                    # E-commerce transaction dataset (~541K records)
│
├── 📦 models/                         # Trained model files
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── gradient_boosting.pkl
│   ├── ridge_regression.pkl
│   ├── linear_regression.pkl
│   └── neural_network.pkl
│
├── 🔐 Encoding & Scaling Files
│   ├── country_encoder.pkl            # LabelEncoder for countries
│   ├── stock_encoder.pkl              # LabelEncoder for product codes
│   └── scaler.pkl                     # StandardScaler for features
│
├── 📈 model_results.json              # Performance metrics for all models
│
├── 📚 Documentation
│   ├── README.md                      # This file
│   ├── BUSINESS_FEATURES_GUIDE.md     # Detailed guide for new features
│   ├── IMPLEMENTATION_SUMMARY.md      # Technical implementation details
│   ├── PROJECT_REPORT.md              # Comprehensive project report
│   └── EDA.ipynb                      # Exploratory Data Analysis notebook
│
├── 📋 requirements.txt                # Python dependencies
└── 💾 saved_model.pkl                # Legacy model file (for reference)
```

---

## 📈 Dataset Information

### Data Source
**E-commerce Transaction Data** - Online retail transactions  
**Records**: 541,909 transactions  
**Columns**: 8 features  
**Time Period**: Multi-year e-commerce data  

### Dataset Features

| Column | Type | Description |
|--------|------|-------------|
| InvoiceNo | String | Unique invoice identifier |
| StockCode | String | Product stock/SKU code |
| Description | String | Product description/name |
| **Quantity** | Integer | **Units purchased (TARGET VARIABLE)** |
| InvoiceDate | DateTime | Transaction date and time |
| UnitPrice | Float | Price per unit (£) |
| CustomerID | Float | Unique customer identifier |
| Country | String | Customer country location |

### Data Quality Metrics
- Missing Values: 1,454 in Description (~0.27%), 135,080 in CustomerID (~24.9%)
- Data Cleaning Applied: Removed negative quantities, invalid prices, and postage items
- Final Clean Dataset: 541,909 records ready for modeling

### Statistical Summary
- **Quantity**: Mean = 9.66 units, Median = 3.0, Range = 1-80,995 units
- **UnitPrice**: Mean = £3.13, Median = £1.95, Range = £0.01-£38,970
- **Geographic Reach**: 37 unique countries (UK dominant at ~91%)
- **Product Catalog**: 4,070 unique products
- **Customer Base**: 4,373 unique customers

---

## 🔧 Technology Stack

### Programming & ML
- **Python 3.8+** - Primary programming language
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Advanced gradient boosting framework
- **NumPy & Pandas** - Data manipulation and analysis
- **TensorFlow/Keras** (via scikit-learn MLPRegressor) - Neural networks

### Data Processing
- **StandardScaler** - Feature normalization
- **LabelEncoder** - Categorical encoding
- **Train-test split** - Data stratification

### Visualization & Dashboard
- **Streamlit** - Interactive web dashboard
- **Plotly** - Interactive charts and visualizations
- **Matplotlib & Seaborn** - Static visualizations
- **Jupyter Notebook** - EDA and analysis

### Data Analysis
- **Pandas** - DataFrames and data analysis
- **NumPy** - Numerical computations
- **SciPy** - Statistical analysis

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone/Download Project
```bash
cd path/to/your/project
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import pandas, numpy, sklearn, xgboost, streamlit; print('All libraries installed successfully!')"
```

---

## 📖 Usage Guide

### Option 1: Train Models & Run Dashboard
```bash
# Train all models (creates model files if not present)
python train_model.py

# Launch Streamlit dashboard
streamlit run app.py
```

### Option 2: Run Dashboard with Pre-trained Models
```bash
# If models are already trained, directly run:
streamlit run app.py
```

### Dashboard Navigation

#### **Tab 1: 🎯 Predictions**
- Select prediction model
- Enter product details (price, country, stock code)
- Get instant demand prediction
- View confidence scores

#### **Tab 2: 📊 Model Comparison**
- Compare all 6 models side-by-side
- View R² scores, MAE, RMSE, MAPE metrics
- Interactive charts for performance comparison
- Identify best-performing model

#### **Tab 3: 📈 Detailed Metrics**
- Detailed analysis for selected model
- Training metrics and cross-validation scores
- Model-specific explanations
- Performance analysis

#### **Tab 4: 💰 Revenue & Profit Impact**
- Financial forecasting based on predictions
- Break-even analysis
- Profit margin calculations
- ROI and inventory metrics
- Financial visualization charts

#### **Tab 5: 🎯 Business Conclusions** (NEW)
- **Alerts System**: Real-time critical notifications
- **Demand Classification**: Automatic categorization
- **Recommendations Engine**: Actionable business suggestions
- **Financial Dashboard**: Revenue, profit, and ROI analysis
- **Inventory Optimization**: Safety stock and reorder points
- **Summary Reports**: Comprehensive analysis table

#### **Tab 6: ℹ️ About**
- Project information
- Technologies overview
- How to use guide
- Model selection recommendations

---

## 🤖 Machine Learning Models

### Model 1: Random Forest Regressor
- **Ensemble method** using multiple decision trees
- **Strengths**: Good for non-linear relationships, provides feature importance
- **Hyperparameters**: 50 estimators, max_depth=15, 4 jobs parallel
- **Best for**: Baseline comparison, feature analysis

### Model 2: XGBoost (Extreme Gradient Boosting)
- **Gradient boosting framework** with optimizations
- **Strengths**: Often best for tabular data, fast training, handles non-linearity
- **Hyperparameters**: 50 estimators, max_depth=5, learning_rate=0.1
- **Best for**: Production deployment, accuracy focus

### Model 3: Gradient Boosting Regressor
- **Sequential ensemble** building trees to correct errors
- **Strengths**: Strong generalization, robust to outliers
- **Hyperparameters**: 50 estimators, max_depth=5
- **Best for**: Stability and reliability

### Model 4: Ridge Regression
- **Regularized linear model** (L2 penalty)
- **Strengths**: Fast, interpretable, prevents overfitting
- **Hyperparameters**: alpha=1.0
- **Best for**: Baseline comparison, explainability

### Model 5: Linear Regression
- **Baseline linear model** without regularization
- **Strengths**: Simple, interpretable, fast
- **Best for**: Understanding basic relationships

### Model 6: Neural Network (MLPRegressor)
- **Deep learning** with hidden layers
- **Strengths**: Captures complex non-linear patterns
- **Hyperparameters**: hidden_layers=(50, 25), max_iter=300, early_stopping=True
- **Best for**: Complex pattern recognition

---

## 📊 Model Performance

### Evaluation Metrics

| Metric | Description | Interpretation |
|--------|-------------|-----------------|
| **R² Score** | Coefficient of determination | Higher is better (1.0 = perfect) |
| **MAE** | Mean Absolute Error | Average prediction error in units |
| **RMSE** | Root Mean Squared Error | Penalizes larger errors more |
| **MAPE** | Mean Absolute % Error | Percentage error (lower is better) |
| **CV Score** | Cross-Validation Score | 5-fold average R² score |

### Evaluation Process
- **Train-Test Split**: 80% training, 20% testing
- **Cross-Validation**: 5-fold CV for robust assessment
- **Scaling**: StandardScaler applied to neural networks
- **Sampling**: Stratified sampling for large datasets (>50K records)

---

## 💡 Key Findings & Insights

### Demand Patterns
- Highly right-skewed quantity distribution (many low-quantity sales)
- Seasonal variations in demand
- Geographic variations in purchasing behavior
- Price elasticity observed in quantity demanded

### Customer Behavior
- Concentration in UK market (~91% of transactions)
- Wide range of purchase patterns (1 to 80,995 units)
- Strong customer segmentation opportunities
- High-value customers identified through RFM analysis

### Product Insights
- Top-selling products differ significantly by country
- Product performance varies by season
- Price point influences quantity demanded
- Product category effects visible in demand patterns

---

## 🎯 Business Applications

### 1. **Inventory Management**
- Optimize stock levels based on demand forecasts
- Reduce holding costs
- Minimize stockout situations
- Improve warehouse efficiency

### 2. **Revenue Optimization**
- Dynamic pricing strategies
- Product bundling recommendations
- Promotional campaign targeting
- Revenue maximization

### 3. **Supply Chain Planning**
- Accurate demand forecasting for suppliers
- Just-in-time inventory strategies
- Procurement optimization
- Lead time adjustments

### 4. **Financial Planning**
- Revenue forecasting
- Profit margin optimization
- Break-even analysis
- ROI calculations

### 5. **Risk Management**
- Stockout risk identification
- Dead stock prediction
- Market change detection
- Demand volatility assessment

---

## 📖 Documentation

### Included Documentation
1. **README.md** (This file) - Complete project overview
2. **BUSINESS_FEATURES_GUIDE.md** - Detailed guide for recommendations engine, alerts, and financial dashboard
3. **IMPLEMENTATION_SUMMARY.md** - Technical implementation details
4. **PROJECT_REPORT.md** - Comprehensive academic project report
5. **EDA.ipynb** - Jupyter notebook with exploratory data analysis

### How to Access
- View in any text editor
- Open Jupyter notebook in Jupyter Lab/Notebook
- View HTML version in browser

---

## 🔍 Advanced Features

### Exploratory Data Analysis (EDA)
Run the included Jupyter notebook for comprehensive data exploration:
```bash
jupyter notebook EDA.ipynb
```

Includes:
- Dataset overview and structure
- Missing values analysis
- Statistical summaries and distributions
- Time-based patterns and seasonality
- Customer and product analysis
- Correlation analysis
- Outlier detection
- Key insights and recommendations

### Business Conclusions Module
Advanced business intelligence features:
- **Intelligent Demand Classification**: VERY_LOW to VERY_HIGH
- **Automatic Recommendations**: Context-aware business suggestions
- **Alert System**: Critical situation detection
- **Financial Analysis**: Revenue, profit, ROI calculations
- **Inventory Optimization**: Safety stock and reorder points

---

## 🎓 Project Competencies Demonstrated

### Technical Skills
- ✅ Machine Learning Model Development
- ✅ Model Comparison and Selection
- ✅ Feature Engineering
- ✅ Data Preprocessing and Cleaning
- ✅ Statistical Analysis
- ✅ Python Programming
- ✅ Scikit-learn and XGBoost expertise

### Data Science Skills
- ✅ Exploratory Data Analysis
- ✅ Cross-validation techniques
- ✅ Hyperparameter optimization
- ✅ Performance metrics evaluation
- ✅ Data visualization
- ✅ Time series analysis

### Business Skills
- ✅ Demand Forecasting
- ✅ Financial Analysis
- ✅ Inventory Management
- ✅ Revenue Optimization
- ✅ Risk Assessment
- ✅ Strategic Decision-Making

### Software Engineering
- ✅ Interactive Dashboard Development (Streamlit)
- ✅ Code Organization and Structure
- ✅ Documentation
- ✅ Model Serialization (joblib)
- ✅ Error Handling

---

## 🚀 Future Enhancements

### Potential Improvements
1. **Deep Learning Models**: LSTM for time series forecasting
2. **Ensemble Methods**: Voting and stacking ensembles
3. **Real-time Data**: Integration with live e-commerce feeds
4. **Advanced NLP**: Description analysis for demand prediction
5. **Automated Retraining**: Pipeline for model updates
6. **Mobile Application**: Mobile dashboard
7. **API Development**: REST API for model serving
8. **Explainability**: SHAP values for model interpretability
9. **Multi-step Forecasting**: Predict demand over multiple periods
10. **Anomaly Detection**: Unusual demand pattern identification

---

## 📞 Support & Resources

### Troubleshooting
- **Models not loading**: Ensure all .pkl files are in the project directory
- **Streamlit errors**: Update Streamlit: `pip install --upgrade streamlit`
- **Memory issues**: Reduce sample size in `train_model.py` for large datasets
- **Import errors**: Verify all packages installed: `pip install -r requirements.txt`

### Reference Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

## 📋 Requirements

### Python Libraries
```
pandas>=1.1.0
numpy>=1.19.0
scikit-learn>=0.24.0
xgboost>=1.3.0
matplotlib>=3.3.0
seaborn>=0.11.0
joblib>=1.0.0
streamlit>=1.0.0
plotly>=4.14.0
scipy>=1.5.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 👥 Author Information

**Project Title**: A STUDY ON AI-BASED PRODUCT DEMAND PREDICTION FOR E-COMMERCE PLATFORMS

**Degree**: Master of Business Administration (MBA)  
**Specialization**: Artificial Intelligence and Data Science  
**Institution**: SRM University  
**Academic Year**: 2024-2026 (Semester 4)  
**Date Completed**: May 2026  

---

## 📄 License & Usage

This project is developed as part of MBA curriculum at SRM University. 

### Usage Terms
- ✅ Educational and research purposes
- ✅ Non-commercial use
- ✅ Attribution appreciated
- ❌ Commercial redistribution without permission

---

## 🏆 Project Highlights

✨ **6 Machine Learning Models** with comparative analysis  
✨ **Intelligent Recommendations Engine** for actionable insights  
✨ **Real-time Alerts System** for critical business situations  
✨ **Comprehensive Financial Dashboard** for revenue optimization  
✨ **Interactive Streamlit Interface** with 6 analysis tabs  
✨ **Complete Documentation** and guides  
✨ **Production-ready Code** with error handling  
✨ **Exploratory Data Analysis** Jupyter notebook  

---

## 📊 Getting Started Checklist

- [ ] Install Python 3.8+
- [ ] Download/clone project files
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Run model training: `python train_model.py`
- [ ] Launch dashboard: `streamlit run app.py`
- [ ] Navigate to business conclusions tab
- [ ] Make predictions and analyze results
- [ ] Review documentation for advanced features
- [ ] Explore EDA notebook for data insights

---

**Last Updated**: May 15, 2026  
**Project Status**: ✅ Complete and Production-Ready  



