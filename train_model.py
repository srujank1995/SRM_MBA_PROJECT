
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import json

# Load dataset with proper encoding
df = pd.read_csv("data_new.csv", encoding='latin-1')

# Remove missing values
df.dropna(inplace=True)

# Remove postage and non-product items
df = df[df['StockCode'] != 'POST']

# Remove negative/zero quantities
df = df[df['Quantity'] > 0]

# Remove cancelled orders (negative prices typically indicate cancellations)
df = df[df['UnitPrice'] > 0]

# Keep valid alphanumeric stock codes but don't be too restrictive
df = df[~df['StockCode'].str.contains(r'[^a-zA-Z0-9]', regex=True, na=False)]

print(f"Dataset Info: {len(df)} records after cleaning")

# Encode categorical variables
country_encoder = LabelEncoder()
stock_encoder = LabelEncoder()

df['Country_Encoded'] = country_encoder.fit_transform(df['Country'])
df['StockCode_Encoded'] = stock_encoder.fit_transform(df['StockCode'])

# Features and target
X = df[['UnitPrice', 'Country_Encoded', 'StockCode_Encoded']]
y = df['Quantity']

# For large datasets, use stratified sampling to speed up training while maintaining data quality
# Take a representative sample if dataset is too large
if len(X) > 50000:
    print(f"Large dataset detected ({len(X)} records). Using stratified sampling for efficiency...")
    sample_size = min(50000, len(X))
    sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
    X = X.iloc[sample_indices]
    y = y.iloc[sample_indices]
    print(f"Using {len(X)} records for training\n")

# Standardize features for neural network and SVR
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_scaled, X_test_scaled, _, _ = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Define all models (optimized for faster training)
models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=50, random_state=42, n_jobs=-1, max_depth=15
    ),
    'XGBoost': XGBRegressor(
        n_estimators=50, random_state=42, verbosity=0, max_depth=5
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=50, random_state=42, max_depth=5
    ),
    'Ridge Regression': Ridge(alpha=1.0),
    'Linear Regression': LinearRegression(),
    'Neural Network': MLPRegressor(
        hidden_layer_sizes=(50, 25), max_iter=300, random_state=42, early_stopping=True
    )
}

# Store results
results = {}
trained_models = {}

print("\n" + "="*70)
print("TRAINING MULTIPLE MODELS FOR DEMAND PREDICTION")
print("="*70 + "\n")

for model_name, model in models.items():
    print(f"Training {model_name}...")
    
    # Use scaled data for Neural Network only
    if model_name == 'Neural Network':
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    
    # Cross-validation score
    if model_name == 'Neural Network':
        cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                   scoring='r2', n_jobs=-1).mean()
    else:
        cv_score = cross_val_score(model, X_train, y_train, cv=5, 
                                   scoring='r2', n_jobs=-1).mean()
    
    # Store results
    results[model_name] = {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2_Score': round(r2, 4),
        'MAPE': round(mape, 2),
        'CV_Score': round(cv_score, 4),
        'Train_Size': len(X_train),
        'Test_Size': len(X_test)
    }
    
    trained_models[model_name] = model
    
    print(f"  ✓ MAE: {mae:.2f}")
    print(f"  ✓ RMSE: {rmse:.2f}")
    print(f"  ✓ R² Score: {r2:.4f}")
    print(f"  ✓ MAPE: {mape:.2f}%")
    print(f"  ✓ CV Score: {cv_score:.4f}\n")

# Display comparison table
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70 + "\n")

results_df = pd.DataFrame(results).T
print(results_df.to_string())

# Find best model by R2 Score
best_model_name = max(results, key=lambda x: results[x]['R2_Score'])
print(f"\n🏆 Best Performing Model: {best_model_name}")
print(f"   R² Score: {results[best_model_name]['R2_Score']}")
print(f"   MAE: {results[best_model_name]['MAE']}")

# Save all models
for model_name, model in trained_models.items():
    filename = f"models/{model_name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, filename)
    print(f"\n✓ Saved: {filename}")

# Save encoders and scaler
joblib.dump(country_encoder, "country_encoder.pkl")
joblib.dump(stock_encoder, "stock_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n✓ Saved: country_encoder.pkl")
print("✓ Saved: stock_encoder.pkl")
print("✓ Saved: scaler.pkl")

# Save results as JSON for dashboard
with open("model_results.json", "w") as f:
    json.dump(results, f, indent=4)
print("✓ Saved: model_results.json")

# Save best model as default
best_model = trained_models[best_model_name]
if best_model_name in ['Support Vector Regressor', 'Neural Network']:
    joblib.dump(best_model, "saved_model.pkl")
else:
    joblib.dump(best_model, "saved_model.pkl")

print("\n" + "="*70)
print("✓ Training Complete! All models and results saved.")
print("="*70 + "\n")

