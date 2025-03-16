from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib
import datetime
import os
import io  # For in-memory file creation

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# ✅ Check if model files exist before loading
if not os.path.exists("doctor_model.pkl") or not os.path.exists("specialty_encoder.pkl"):
    raise FileNotFoundError("Model files not found. Ensure doctor_model.pkl and specialty_encoder.pkl exist in the directory.")

# Load ML model and label encoder
model = joblib.load("doctor_model.pkl")
label_encoder = joblib.load("specialty_encoder.pkl")

# ✅ Ensure dataset file exists
dataset_path = "data.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file '{dataset_path}' not found. Place it in the same directory as app.py.")

df = pd.read_csv(dataset_path)

# ✅ Preprocess dataset
df["Login Hour"] = pd.to_datetime(df["Login Time"], errors='coerce').dt.hour
df = df.dropna(subset=["Login Hour"])  # Remove rows with invalid Login Time
df["Speciality"] = label_encoder.transform(df["Speciality"])

# ✅ Home route to prevent 404 errors
@app.route("/")
def serve_html():
    return send_file(".", "index.html")  # Serving from the current directory

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({"message": "Use POST with time data"}), 200

    try:
        data = request.json
        print("Received request data:", data)

        input_time = data.get("time", "").strip()
        if not input_time:
            return jsonify({"error": "Time is required"}), 400

        try:
            input_hour = datetime.datetime.strptime(input_time, "%H:%M").hour
            print(f"✅ Parsed hour: {input_hour}")
        except ValueError:
            return jsonify({"error": "Invalid time format. Use HH:MM."}), 400

        print("Columns in dataset after processing:", df.columns)

        # Ensure 'Login Hour' exists
        if "Login Hour" not in df.columns:
            df["Login Hour"] = pd.to_datetime(df["Login Time"], errors='coerce').dt.hour
            df.dropna(subset=["Login Hour"], inplace=True)

        doctors_active = df[df["Login Hour"] == input_hour]
        print(f"✅ Doctors found at {input_hour}: {len(doctors_active)}")

        if doctors_active.empty:
            return jsonify([])

        features = doctors_active[["Login Hour", "Usage Time (mins)", "Count of Survey Attempts", "Speciality"]]
        print("Feature data being passed to model:\n", features.head())

        # Check for missing values
        if features.isnull().values.any():
            print("❌ Error: Missing values detected in features!")
            return jsonify({"error": "Missing values in data"}), 500

        predictions = model.predict(features)
        print("✅ Model Predictions:", predictions)

        doctors_active = doctors_active.assign(Prediction=predictions)
        final_list = doctors_active[doctors_active["Prediction"] == 1]
        npis_to_export = final_list["NPI"].tolist()  # Extract NPIs only

        # Create a Pandas DataFrame with only NPIs
        df_export = pd.DataFrame({'NPI': npis_to_export})

        # Create an in-memory Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_export.to_excel(writer, sheet_name='NPIs', index=False)
        output.seek(0)

        # Send the Excel file as a downloadable response
        if request.headers.get('Accept') == 'application/json':
            return jsonify(npis_to_export)
        else :
            return send_file(output, as_attachment=True, download_name='predicted_npis.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
