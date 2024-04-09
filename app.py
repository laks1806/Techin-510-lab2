import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")

# Set page configuration
st.set_page_config(
    page_title="Boston Housing Predictor",
    page_icon="🏠",
    layout="centered"
)

# Title and description
st.title("Boston Housing Predictor 🏠")
# st.markdown("""
# Predict the median value of owner-occupied homes in Boston using a linear regression model.
# """)

# Sidebar for user input
st.sidebar.header('Adjust Parameters')

# Get user inputs for features
features = {}
for feature in df.columns[:-1]:  # Exclude target variable (MEDV)
    features[feature] = st.sidebar.slider(
        f'{feature}',
        float(df[feature].min()),
        float(df[feature].max()),
        float(df[feature].mean())
    )

# Create feature vector
X = pd.DataFrame(features, index=[0])

# Prediction function
def predict_price(X):
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=['medv']),
        df['medv'],
        test_size=0.2,
        random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X)
    return prediction[0]

# Make prediction
prediction = predict_price(X)

# Display prediction
st.markdown("### Predicted Median Value of Owner-Occupied Homes(in 1000$)")
predicted_value = f"${prediction:,.2f}"
st.info(predicted_value)
st.markdown(f"<p style='text-align: center;'><span style='font-size: 18px;'>🔄 Change the sliders on the left to see the prediction update 🔄</span></p>", unsafe_allow_html=True)


# Scatter plot of Crime Rate vs Median Home Values
st.subheader('Scatter Plot of Crime Rate vs. Median Home Value (MEDV)')
st.markdown("""
We can observe whether there's a relationship between crime rate and median home value. A higher crime rate might lead to lower home values.
""")
fig_scatter = px.scatter(df, x='crim', y='medv', title='Crime Rate vs Median Home Values', hover_data=['crim', 'medv'])
st.plotly_chart(fig_scatter)

# Correlation Heatmap
st.subheader('Correlation Heatmap')
st.markdown("""
A heatmap showing the correlation matrix between all features and MEDV. We can identify features that have strong positive or negative correlations with median home value. This can guide us in understanding which factors are most influential in determining home values.
""")
corr = df.corr()
fig_corr = px.imshow(corr, text_auto=True, title='Correlation Heatmap', aspect="auto")
st.plotly_chart(fig_corr)

# List of observations
observations = [
    "The target variable MEDV has a strong positive correlation with RM (Average number of rooms per dwelling), indicating that homes with more rooms tend to have higher median values.",
    "MEDV also has a moderate positive correlation with LSTAT (Percentage of lower status population), suggesting that areas with a higher percentage of lower-status population tend to have lower median home values.",
    "MEDV shows a moderate negative correlation with CRIM (Per capita crime rate by town), implying that areas with higher crime rates tend to have lower median home values.",
    "There is a moderate negative correlation between MEDV and AGE (Proportion of owner-occupied units built before 1940), indicating that older homes tend to have lower median values.",
    "The feature RAD (Index of accessibility to radial highways) has a moderate negative correlation with MEDV, suggesting that homes farther away from radial highways tend to have lower median values.",
    "INDUS (Proportion of non-retail business acres per town) and NOX (Nitric oxides concentration) both have moderate negative correlations with MEDV, indicating that higher levels of these features are associated with lower median home values.",
    "The correlation between MEDV and other features like ZN (Proportion of residential land zoned for lots over 25,000 sq.ft.), CHAS (Charles River dummy variable), TAX (Full-value property-tax rate per $10,000), PTRATIO (Pupil-teacher ratio by town), and B (Proportion of blacks by town) is relatively weaker.",
    "There are strong positive correlations among some features like RM and LSTAT, and strong negative correlations among others like CRIM and AGE, suggesting potential multicollinearity issues."
]

st.markdown("Based on the correlation heatmap, we can make the following observations and understand the relationships between the features and the target variable (MEDV - Median value of owner-occupied homes):")
for i, obs in enumerate(observations, start=1):
    st.markdown(f"{i}. {obs}")