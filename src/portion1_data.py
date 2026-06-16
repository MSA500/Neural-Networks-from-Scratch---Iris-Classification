import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

import os
os.makedirs('outputs', exist_ok=True)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['species'] = [iris.target_names[t] for t in iris.target]

print("Dataset Shape  :", df.shape)
print("Classes        :", df['species'].unique())
print("Class Counts   :\n", df['species'].value_counts())
print("\nFirst 5 rows:\n", df.head())
print("\nStatistics:\n", df.describe().round(3))
print("\nMissing Values :", df.isnull().sum().sum())

df['label'] = (df['species'] == 'setosa').astype(int)

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("\nTrain samples  :", X_train_sc.shape[0])
print("Test  samples  :", X_test_sc.shape[0])
print("Train balance  :", np.bincount(y_train))
print("Test  balance  :", np.bincount(y_test))

colors_map = {'setosa': "#4d3ce7", 'versicolor': "#FFFB00", 'virginica': "#000000"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Iris Dataset — Feature Distribution', fontsize=13, fontweight='bold')

for species, color in colors_map.items():
    mask = df['species'] == species
    axes[0].scatter(df[mask]['sepal_length'], df[mask]['sepal_width'],
                    c=color, label=species, alpha=0.75, edgecolors='k', linewidth=0.4, s=65)
axes[0].set_xlabel('Sepal Width (cm)')
axes[0].set_ylabel('Sepal Length (cm)')
axes[0].set_title('Sepal: Length vs Width')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

for species, color in colors_map.items():
    mask = df['species'] == species
    axes[1].scatter(df[mask]['petal_length'], df[mask]['petal_width'],
                    c=color, label=species, alpha=0.75, edgecolors='k', linewidth=0.4, s=65)
axes[1].set_xlabel('Petal Width (cm)')
axes[1].set_ylabel('Petal Length (cm)')
axes[1].set_title('Petal: Length vs Width')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/portion1_exploration.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to outputs/portion1_exploration.png")

np.save('outputs/X_train.npy', X_train_sc)
np.save('outputs/X_test.npy',  X_test_sc)
np.save('outputs/y_train.npy', y_train)
np.save('outputs/y_test.npy',  y_test)
print("Preprocessed data saved to outputs/")
