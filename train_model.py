from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Loading data
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
