
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# טעינת המודלים והרשימות
reg_model = joblib.load('parking_spots_predictor_reg_model.pkl')
reg_features = joblib.load('parking_spots_reg_model_features_list.pkl')
clf_model = joblib.load('parking_predictor_model.pkl')
clf_features = joblib.load('main_clf_model_features_list.pkl')

# טעינת האנקודרים
day_encoder = joblib.load('day_of_week_encoder.pkl')
time_encoder = joblib.load('time_of_day_encoder.pkl')
city_encoder = joblib.load('city_encoder.pkl')
parking_encoder = joblib.load('parking_name_encoder.pkl')

# קריאה לממשק /predict
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # קידוד קלטים
        day_encoded = day_encoder.transform([data['day_of_week']])[0]
        time_encoded = time_encoder.transform([data['time_of_day']])[0]

        city_onehot = city_encoder.transform([[data['city']]]).flatten()
        parking_onehot = parking_encoder.transform([[data['parking_name']]]).flatten()

        reg_input_dict = {
            'encoded_day_of_week': day_encoded,
            'encoded_time_of_day': time_encoded,
            'combined_parking_cost_status': data['combined_parking_cost_status'],
            'encoded_abnormal_parking': data['encoded_abnormal_parking'],
            'encoded_parking_type': data['parking_type_encoded'],
            'טמפ': data['temperature_c'],
            'כמות חניה סהכ': data['total_parking_spots']
        }

        reg_input = list(reg_input_dict.values()) + list(city_onehot) + list(parking_onehot)
        reg_df = pd.DataFrame([reg_input], columns=reg_features)

        predicted_spots = max(0, round(reg_model.predict(reg_df)[0]))

        clf_input_dict = reg_input_dict.copy()
        clf_input_dict['parking_spots_available_current'] = predicted_spots

        clf_input = list(clf_input_dict.values()) + list(city_onehot) + list(parking_onehot)
        clf_df = pd.DataFrame([clf_input], columns=clf_features)

        prediction = clf_model.predict(clf_df)[0]
        proba = clf_model.predict_proba(clf_df)[0]

        return jsonify({
            'predicted_is_available': int(prediction),
            'predicted_available_spots': int(predicted_spots),
            'probability_available': round(proba[1], 3),
            'probability_not_available': round(proba[0], 3)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
