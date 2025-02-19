# Iris Classification App with Streamlit

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['species'] = iris.target_names[iris.target]

# App Title
st.title("Iris Classification App")
st.write("The \"Hello World\" of Machine Learning!")

# Sidebar
st.sidebar.title("About the Project") 
st.sidebar.write("This app predicts the **species** of the Iris flower based on the **sepal length**, **sepal width**, **petal length**, and **petal width**.")
st.sidebar.write("The Iris dataset consists of 150 samples of Iris flowers, each with 4 features.")
st.sidebar.write("The task is to classify the Iris flowers into one of the three species: **setosa**, **versicolor**, or **virginica**.")
st.sidebar.write("The app uses a **Random Forest Classifier** to predict the species of the Iris flower.")
st.sidebar.write("The model is trained on 80% of the dataset and tested on the remaining 20%.")
st.sidebar.write("Note: I'm career transitioning to Data Science lol.")

# Display the dataset
st.subheader("Dataset")
# Show the whole dataset on user's request (toggle)
if st.checkbox("Show Whole Dataset"):   
    st.write(iris_df)
else:   
    st.write(iris_df.head())

# Summary Statistics
st.subheader("Summary Statistics")
st.write(iris_df.describe())

# Data Visualization
st.subheader("Data Visualization")

# Pairplot
st.write("Pairplot of the Iris Dataset")
pairplot = sns.pairplot(iris_df, hue='species')
st.pyplot(pairplot)

# Correlation Matrix
st.write("Correlation Matrix")
corr_matrix = iris_df.corr()
st.write(corr_matrix)

# Heatmap of Correlation Matrix
st.write("Heatmap of Correlation Matrix")
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
st.pyplot(plt)

# Machine Learning Model
st.subheader("Machine Learning Model")

# Split the data
X = iris_df.drop(['target', 'species'], axis=1)
y = iris_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model Evaluation
st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))