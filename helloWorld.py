import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt


df = pd.read_csv('data.csv')

# Convert categorical data to numerical data
df['Nationality'] = df['Nationality'].map({'UK': 0, 'USA': 1, 'N': 2})
df['Go'] = df['Go'].map({'YES': 1, 'NO': 0})

# Define features and target
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']

# Create and train the decision tree classifier
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(dtree, feature_names=features, class_names=['NO', 'YES'], filled=True)
plt.show()