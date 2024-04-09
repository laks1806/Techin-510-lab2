import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# df = pd.DataFrame(boston.data, columns=boston.feature_names)
df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")

# Set page config
st.set_page_config(
    page_title="Boston Housing Predictor",
    page_icon="🏠",
    layout="centered"
)

# Title and description
st.title("Boston Housing Predictor")
st.markdown("""
Predict the median value of owner-occupied homes in Boston using a linear regression model.
""")

# Sidebar for user input
st.sidebar.header('Adjust Parameters')

# User inputs
features = {}
for feature in df.columns[:-1]:  # Exclude target variable (MEDV)
    features[feature] = st.sidebar.slider(f'{feature}', float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))

# Create feature vector
X = pd.DataFrame(features, index=[0])

# Prediction function
def predict_price(X):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['medv']), df['medv'], test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X)
    return prediction[0]

# Predict
prediction = predict_price(X)

# Display prediction
st.subheader('Predicted Median Value of Owner-Occupied Homes in $1000s')
st.write(f"${prediction:,.2f}")

st.subheader('Histogram of Median Home Values')
fig_hist = px.histogram(df, x='medv', title='Distribution of Median Home Values')
st.plotly_chart(fig_hist)

# Scatter plot
st.subheader('Scatter Plot of Rooms vs Median Home Values')
fig_scatter = px.scatter(df, x='rm', y='medv', title='Rooms vs Median Home Values')
st.plotly_chart(fig_scatter)