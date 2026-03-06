from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

# Load dataset to populate dropdown
df = pd.read_csv("notebook/Data/data.csv", encoding="latin1")

countries = sorted(df["Country"].dropna().unique())


@app.route('/')
def home():
    return render_template("Home.html")


@app.route('/predict', methods=["GET","POST"])
def predict():

    if request.method == "POST":

        country = request.form.get("Country")

        data = pd.DataFrame({
            "Country":[country]
        })

        pipeline = PredictPipeline()

        result = pipeline.predict(data)

        return render_template(
            "index.html",
            countries=countries,
            results=round(result[0],2)
        )

    return render_template("index.html", countries=countries)


if __name__ == "__main__":
    app.run(debug=True)