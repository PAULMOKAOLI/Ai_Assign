import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model and transformers
with open('ai_model10.pkl', 'rb') as file:
    model = pickle.load(file)

with open('level_of_government_encoder.pkl', 'rb') as file:
    level_of_government_encoder = pickle.load(file)

with open('one_hot_encoder.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the revenue category mapping
revenue_category_mapping = {
    1: 'Total tax and non-tax revenue', 
    2: 'Total tax revenue',
    3: '1000 Taxes on income, profits and capital gains',
    4: '1110 On income and profits of individuals',
    5: 'Personal income tax',
    6: 'Withholding tax',
    7: '1120 On capital gains of individuals',
    8: '1200 Taxes on income, profits and capital gains of corporates',
    9: '1210 On profits of corporates',
    10: 'Corporate income tax',
    11: 'Gambling levy',
    12: '1220 On capital gains of corporates',
    13: '1300 Unallocable between 1100 and 1200',
    14: '2000 Social security contributions (SSC)',
    15: '3000 Taxes on payroll and workforce',
    16: '5000 Taxes on goods and services',
    17: '5100 Taxes on production, sale, transfer, etc',
    18: '5110 General taxes on goods and services',
    19: '5111 Value added taxes',
    20: 'VAT on imports',
    21: 'VAT - Domestic (Net)',
    22: '5112 Sales tax',
    23: '5113 Other (than value added and sales tax)',
    24: '5120 Taxes on specific goods and services',
    25: '5130 Unallocable between 5110 and 5120',
    26: '5200 Taxes on use of goods and perform activities',
    27: '5300 Unallocable between 5100 and 5200',
    28: '6000 Taxes other than 1000, 2000, 3000, 4000 and 5000',
    29: 'Total tax revenues not including social security contributions',
    30: 'Total non-tax revenue',
    31: 'Non-tax revenue: Grants',
    32: 'Non-tax revenue: Rents and royalties',
    33: 'Non-tax revenue: Property income',
    34: 'Water royalties',
    35: 'Mining and other royalties',
    36: 'Non-tax revenue: Interest and dividends',
    37: 'Non-tax revenue: Other property income',
    38: 'Non-tax revenue: Sales of goods and services',
    39: 'Non-tax revenue: Miscellaneous and unidentified revenue',
    40: 'SACU revenue',
    41: 'Total non-tax revenue excluding grants',
    42: 'Total tax and non-tax revenue excluding grants',
    43: 'Excise taxes collected on behalf of the SACU Common Revenue Pool',
    44: 'Import duties collected on behalf of the SACU Common Revenue Pool',
    45: 'VAT - Domestic (Gross)',
    46: 'VAT - Domestic (Refunds)'
}

# Reverse the mapping for displaying in the selectbox
revenue_category_options = {v: k for k, v in revenue_category_mapping.items()}

# Define the app
st.title("Tax Revenue Prediction App")

# Input fields
level_of_government_options = level_of_government_encoder.classes_
level_of_government = st.selectbox("Select Level of Government", level_of_government_options)
revenue_category = st.selectbox("Select Revenue Category", list(revenue_category_options.keys()))
year = st.number_input("Enter Year", min_value=2001, max_value=2024, step=1)

# Predict button
if st.button("Predict"):
    # Encode inputs
    level_of_government_encoded = level_of_government_encoder.transform([level_of_government])[0]
    revenue_category_encoded = revenue_category_options[revenue_category]

    # Create input dataframe
    input_data = pd.DataFrame({
        'Level of government': [level_of_government_encoded],
        'Revenue category': [revenue_category_encoded],
        'Year': [year]
    })

    # One-hot encode categorical features
    encoded_features = one_hot_encoder.transform(input_data[['Level of government', 'Revenue category']])
    
    # Scale the 'Year' column
    input_data[['Year']] = scaler.transform(input_data[['Year']])
    
    # Combine encoded features with scaled year
    input_features = np.concatenate((input_data[['Year']], encoded_features), axis=1)

    # Make prediction
    prediction = model.predict(input_features) * -1

    # Display the prediction
    st.write(f"Predicted Revenue: {prediction[0]:.2f}")