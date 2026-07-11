from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# Load the saved pipeline
pipeline = joblib.load("xgb_fraud_pipeline.joblib")

# Define expected input schema
class Transaction(BaseModel):
    amt: float
    lat: float
    city_pop: int
    unix_time: int
    merch_lat: float
    trans_hour: int
    age: int
    gender_M: int
    category_food_dining: int
    category_gas_transport: int
    category_grocery_net: int
    category_grocery_pos: int
    category_health_fitness: int
    category_home: int
    category_kids_pets: int
    category_misc_net: int
    category_misc_pos: int
    category_personal_care: int
    category_shopping_net: int
    category_shopping_pos: int
    category_travel: int
    trans_day_of_week: int
    is_weekend: int
    trans_month: int
    log_amt: float
    distance_km: float
    age_group_18_30: int = Field(..., alias="age_group_18-30")
    age_group_30_45: int = Field(..., alias="age_group_30-45")
    age_group_45_60: int = Field(..., alias="age_group_45-60")
    age_group_60_plus: int = Field(..., alias="age_group_60+")

    class Config:
        allow_population_by_field_name = True  # allow internal names to be used in code

# Create FastAPI instance
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API Running "}

@app.post("/predict")
def predict_fraud(data: Transaction):
    # Convert to DataFrame using internal field names
    input_dict = {
        "amt": data.amt,
        "lat": data.lat,
        "city_pop": data.city_pop,
        "unix_time": data.unix_time,
        "merch_lat": data.merch_lat,
        "trans_hour": data.trans_hour,
        "age": data.age,
        "gender_M": data.gender_M,
        "category_food_dining": data.category_food_dining,
        "category_gas_transport": data.category_gas_transport,
        "category_grocery_net": data.category_grocery_net,
        "category_grocery_pos": data.category_grocery_pos,
        "category_health_fitness": data.category_health_fitness,
        "category_home": data.category_home,
        "category_kids_pets": data.category_kids_pets,
        "category_misc_net": data.category_misc_net,
        "category_misc_pos": data.category_misc_pos,
        "category_personal_care": data.category_personal_care,
        "category_shopping_net": data.category_shopping_net,
        "category_shopping_pos": data.category_shopping_pos,
        "category_travel": data.category_travel,
        "trans_day_of_week": data.trans_day_of_week,
        "is_weekend": data.is_weekend,
        "trans_month": data.trans_month,
        "log_amt": data.log_amt,
        "distance_km": data.distance_km,
        "age_group_18-30": data.age_group_18_30,
        "age_group_30-45": data.age_group_30_45,
        "age_group_45-60": data.age_group_45_60,
        "age_group_60+": data.age_group_60_plus
    }

    input_df = pd.DataFrame([input_dict])

    # Predict
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    return {
    "prediction": int(prediction),
    "fraud_probability": float(probability)  # Ensures it's a Python float, not numpy.float32
}

