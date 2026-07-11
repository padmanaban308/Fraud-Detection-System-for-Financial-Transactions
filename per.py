import matplotlib.pyplot as plt
import pandas as pd
import joblib

# Load your saved pipeline or model
pipeline = joblib.load("xgb_fraud_pipeline.joblib")

# If your pipeline includes preprocessing steps, 
# extract the final XGBClassifier model
# (assuming pipeline is sklearn Pipeline)
xgb_model = None
try:
    xgb_model = pipeline.named_steps['xgbclassifier']
except:
    # If pipeline is just the model itself:
    xgb_model = pipeline

# Get feature importances from the model
importances = xgb_model.feature_importances_

# If you have a list of feature names (important!):
feature_names = [
    "amt", "lat", "city_pop", "unix_time", "merch_lat", "trans_hour", "age", "gender_M",
    "category_food_dining", "category_gas_transport", "category_grocery_net", "category_grocery_pos",
    "category_health_fitness", "category_home", "category_kids_pets", "category_misc_net",
    "category_misc_pos", "category_personal_care", "category_shopping_net", "category_shopping_pos",
    "category_travel", "trans_day_of_week", "is_weekend", "trans_month", "log_amt", "distance_km",
    "age_group_18-30", "age_group_30-45", "age_group_45-60", "age_group_60+"
]

# Create a DataFrame to view importances
feat_imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(feat_imp_df)

# Optional: plot feature importances
plt.figure(figsize=(10,6))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importances from XGBoost Model")
plt.show()
