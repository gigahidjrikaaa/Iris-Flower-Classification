# Iris Classification App with Streamlit

import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# Function to predict species using Euclidean distance
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    # Calculate the Euclidean distance between the input features and each sample in the dataset
    iris_df['distance'] = iris_df.apply(lambda row: euclidean_distance(
        [sepal_length, sepal_width, petal_length, petal_width],
        [row['sepal length (cm)'], row['sepal width (cm)'], row['petal length (cm)'], row['petal width (cm)']]
    ), axis=1)
    # Get the species of the sample with the minimum distance   
    species = iris_df.loc[iris_df['distance'].idxmin()]['species']
    return species

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
st.write("To visualize the data, we will use scatter plots to show the relationship between the features.")
st.write("The scatter plots show the relationship between the following features:")
st.write("- Sepal Length vs Sepal Width")
st.write("- Sepal Length vs Petal Length")
st.write("- Petal Length vs Petal Width")
st.write("- Sepal Width vs Petal Width")
st.write("The points are colored based on the species of the Iris flower.")

col1, col2 = st.columns(2)

with col1:
    # Scatter Plot for Sepal Length vs Sepal Width
    fig, ax = plt.subplots()
    sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', data=iris_df, ax=ax)
    ax.set_title("Sepal Length vs Sepal Width")
    st.pyplot(fig)

    # Scatter Plot for Sepal Length vs Petal Length
    fig, ax = plt.subplots()
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=iris_df, ax=ax)
    ax.set_title("Sepal Length vs Petal Length")
    st.pyplot(fig)

with col2:
    # Scatter Plot for Petal Length vs Petal Width
    fig, ax = plt.subplots()
    sns.scatterplot(x='petal length (cm)', y='petal width (cm)', hue='species', data=iris_df, ax=ax)
    ax.set_title("Petal Length vs Petal Width")
    st.pyplot(fig)

    # Scatter Plot for Sepal Width vs Petal Width
    fig, ax = plt.subplots()
    sns.scatterplot(x='sepal width (cm)', y='petal width (cm)', hue='species', data=iris_df, ax=ax)
    ax.set_title("Sepal Width vs Petal Width")
    st.pyplot(fig)


# Input fields for user to enter data
st.header("Predict the Species of the Iris Flower")
st.write("Enter the values of the following features to predict the species of the Iris flower.")

st.subheader("Input Features")
sepal_length = st.slider("Sepal Length", float(iris_df['sepal length (cm)'].min()), float(iris_df['sepal length (cm)'].max()))  
sepal_width = st.slider("Sepal Width", float(iris_df['sepal width (cm)'].min()), float(iris_df['sepal width (cm)'].max()))
petal_length = st.slider("Petal Length", float(iris_df['petal length (cm)'].min()), float(iris_df['petal length (cm)'].max()))
petal_width = st.slider("Petal Width", float(iris_df['petal width (cm)'].min()), float(iris_df['petal width (cm)'].max()))


# Predict the species
species = predict_species(sepal_length, sepal_width, petal_length, petal_width)

# Display the prediction
st.subheader("Prediction")
st.write("The species of the Iris flower is:", species)
if species == 'setosa':
    st.image("setosa.jpg", width=200)
elif species == 'versicolor':
    st.image("versicolor.jpg", width=200)
else:
    st.image("virginica.jpg", width=200)