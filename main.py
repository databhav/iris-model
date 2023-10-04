import streamlit as st
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
# Title
st.title('Iris Dataset - Actual vs Predicted')

# Load Iris dataset
iris = sklearn.datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Sidebar - User Input
st.sidebar.subheader('Select Model Hyperparameters')
split_ratio = st.sidebar.slider('Train-Test Split Ratio', 0.1, 1.0, 0.8)
n_estimators = st.sidebar.slider('Number of Estimators (trees)', 1, 100, 10)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

# Build Random Forest Classifier
model = RandomForestClassifier(n_estimators=n_estimators)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


col1, col2 = st.columns((1,1))
with col1:
  # Display accuracy
  st.write('Accuracy:', accuracy_score(y_test, y_pred))

  # Create a DataFrame for actual vs predicted values
  df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

  # Display actual vs predicted values
  st.write('Actual vs Predicted Values:')
  st.write(df)

with col2:
  # Plot actual vs predicted
  fig, ax = plt.subplots()
  ax.scatter(y_test, y_pred)
  ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
  ax.set_xlabel('Actual')
  ax.set_ylabel('Predicted')
  st.pyplot(fig)

iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a K-nearest neighbors classifier model
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

# Evaluating the precision of the model
accuracy = clf.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")
