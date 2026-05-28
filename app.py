from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import numpy as np
import pandas as pd


app = Flask(__name__)
CORS(app)


model = xgb.XGBRegressor()
model.load_model("ipl_model.json")  


expected_features = model.get_booster().feature_names
print("✅ Model expects features:", expected_features)


teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Kolkata Knight Riders',
    'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]


def preprocess_input(data):
    batting_team = data['batting_team']
    bowling_team = data['bowling_team']
    runs = data['runs']
    wickets = data['wickets']
    overs = data['overs']

    
    input_dict = {f'batting_team_{team}': 0 for team in teams}
    input_dict.update({f'bowling_team_{team}': 0 for team in teams})

    input_dict[f'batting_team_{batting_team}'] = 1
    input_dict[f'bowling_team_{bowling_team}'] = 1

   
    input_dict['runs'] = runs
    input_dict['wickets'] = wickets
    input_dict['overs'] = overs
    input_dict['runs_per_over'] = runs / overs if overs != 0 else 0

    
    df = pd.DataFrame([input_dict])
    df = df.reindex(columns=expected_features, fill_value=0)  

    return df


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("📥 Received input:", data)

        input_df = preprocess_input(data)
        print("🧾 Input DataFrame:\n", input_df)

        prediction = model.predict(input_df)[0]
        print("📈 Predicted Score:", prediction)

        return jsonify({'predicted_score': round(prediction)})

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)

