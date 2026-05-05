from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from src.utils import load_object, evaluate_models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor

app = Flask(__name__)

# Load dataset to populate dropdown
df = pd.read_csv("notebook/data/data.csv", encoding="latin1")

countries = sorted(df["Country"].dropna().unique())


def _get_dashboard_data():
    """Compute model metrics and per-country stats from training data."""

    train = pd.read_csv("artifacts/train.csv")
    test  = pd.read_csv("artifacts/test.csv")

    target = "Quantity"
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test  = test.drop(columns=[target])
    y_test  = test[target]

    preprocessor = load_object("artifacts/preprocessor.pkl")
    X_train_arr = preprocessor.transform(X_train)
    X_test_arr  = preprocessor.transform(X_test)

    models = {
        "Random Forest":      RandomForestRegressor(),
        "Linear Regression":  LinearRegression(),
        "CatBoost":           CatBoostRegressor(verbose=False),
    }

    report_test  = evaluate_models(X_train_arr, y_train, X_test_arr,  y_test,  models)
    report_train = evaluate_models(X_train_arr, y_train, X_train_arr, y_train, models)

    best_name = max(report_test, key=report_test.get)

    model_rows = []
    for name in models:
        model_rows.append({
            "name":        name,
            "train_score": report_train[name],
            "test_score":  report_test[name],
            "is_best":     name == best_name,
        })

    # Sort: best model first, then by descending test R²
    model_rows.sort(key=lambda x: (not x["is_best"], -x["test_score"]))

    # Per-country stats from training data
    by_country = (
        train.groupby("Country")["Quantity"]
        .agg(TransactionCount="count", TotalQuantity="sum",
             AvgQuantity="mean", MinQuantity="min", MaxQuantity="max")
        .reset_index()
    )
    total_qty = by_country["TotalQuantity"].sum()
    by_country["share_pct"] = (by_country["TotalQuantity"] / total_qty * 100).round(4)
    by_country = by_country.sort_values("TotalQuantity", ascending=False)
    country_stats = by_country.to_dict(orient="records")

    summary = {
        "total_rows":   len(train),
        "countries":    train["Country"].nunique(),
        "avg_quantity": float(train["Quantity"].mean()),
        "total_quantity": int(train["Quantity"].sum()),
    }

    return model_rows, country_stats, summary


@app.route('/')
def home():
    return render_template("Home.html")


@app.route('/dashboard')
def dashboard():
    model_rows, country_stats, summary = _get_dashboard_data()
    return render_template(
        "dashboard.html",
        models=model_rows,
        country_stats=country_stats,
        summary=summary,
    )


@app.route('/predict', methods=["GET", "POST"])
def predict():

    results = None
    selected_country = None

    if request.method == "POST":

        country = request.form.get("Country")
        selected_country = country

        data = pd.DataFrame({"Country": [country]})

        pipeline = PredictPipeline()
        result = pipeline.predict(data)
        results = round(result[0], 2)

    return render_template(
        "index.html",
        countries=countries,
        results=results,
        selected_country=selected_country,
    )


if __name__ == "__main__":
    app.run(debug=True)