import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DeltaRuleGD:
    def __init__(self, learning_rate=0.01, n_epochs=100):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_deriv(self, a):
        return a * (1 - a)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_epochs):
            total_loss = 0.0

            for xi, yi in zip(X, y):
                z = np.dot(xi, self.weights) + self.bias
                out = self._sigmoid(z)
                err = yi - out

                loss = 0.5 * (err ** 2)
                total_loss += loss

                dLoss_dout = -err
                dout_dz = self._sigmoid_deriv(out)
                dLoss_dz = dLoss_dout * dout_dz

                for j in range(n_features):
                    self.weights[j] -= self.lr * dLoss_dz * xi[j]

                self.bias -= self.lr * dLoss_dz

            mse = total_loss / n_samples
            self.loss_history.append(mse)

        return self

    def predict(self, X):
        out = self._sigmoid(self._net_input(X))
        return (out >= 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


X_train = np.load('outputs/X_train.npy')
X_test  = np.load('outputs/X_test.npy')
y_train = np.load('outputs/y_train.npy')
y_test  = np.load('outputs/y_test.npy')

model = DeltaRuleGD(learning_rate=0.01, n_epochs=100)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc  = model.score(X_test,  y_test)

print("Gradient Descent Delta Rule")
print("Learning Rate :", model.lr)
print("Epochs        :", model.n_epochs)
print("Weights       :", model.weights.round(4))
print("Bias          :", round(model.bias, 4))
print("Final MSE     :", round(model.loss_history[-1], 6))
print("Train Accuracy:", f"{train_acc * 100:.2f}%")
print("Test  Accuracy:", f"{test_acc  * 100:.2f}%")

y_pred = model.predict(X_test)
TP = np.sum((y_pred == 1) & (y_test == 1))
TN = np.sum((y_pred == 0) & (y_test == 0))
FP = np.sum((y_pred == 1) & (y_test == 0))
FN = np.sum((y_pred == 0) & (y_test == 1))

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\nConfusion Matrix (Test Set):")
print(f"  True Neg : {TN}   False Pos: {FP}")
print(f"  False Neg: {FN}   True Pos : {TP}")
print(f"\nPrecision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle('Gradient Descent Delta Rule — Results', fontsize=13, fontweight='bold')

axes[0].plot(range(1, model.n_epochs + 1), model.loss_history,
             color='#9b59b6', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Loss per Epoch')
axes[0].grid(alpha=0.3)

cats = ['Train Accuracy', 'Test Accuracy']
vals = [train_acc * 100, test_acc * 100]
bars = axes[1].bar(cats, vals, color=['#3498db', '#e67e22'],
                   edgecolor='k', linewidth=0.8, width=0.4)
for bar, v in zip(bars, vals):
    axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                 f'{v:.1f}%', ha='center', fontsize=11)
axes[1].set_ylim(0, 115)
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Train vs Test Accuracy')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/portion3_delta.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to outputs/portion3_delta.png")

np.save('outputs/gd_loss.npy', np.array(model.loss_history))
