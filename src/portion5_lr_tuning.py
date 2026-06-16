import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_d(a):
    return a * (1 - a)


class GDModel:
    def __init__(self, lr=0.01, n_epochs=100):
        self.lr = lr
        self.n_epochs = n_epochs
        self.loss = []
        self.acc  = []

    def fit(self, X, y):
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        for _ in range(self.n_epochs):
            z   = X @ w + b
            out = sigmoid(z)
            err = y - out
            w  += self.lr * (X.T @ (err * sigmoid_d(out))) / n
            b  += self.lr * (err * sigmoid_d(out)).mean()
            self.loss.append(0.5 * np.mean(err ** 2))
            self.acc.append(np.mean(((out >= 0.5).astype(int)) == y))
        self.weights = w
        self.bias    = b
        return self

    def predict(self, X):
        return (sigmoid(X @ self.weights + self.bias) >= 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class PerceptronModel:
    def __init__(self, lr=0.1, n_epochs=50):
        self.lr = lr
        self.n_epochs = n_epochs
        self.errors = []

    def fit(self, X, y):
        w = np.zeros(X.shape[1])
        b = 0.0
        for _ in range(self.n_epochs):
            e = 0
            for xi, yi in zip(X, y):
                z  = np.dot(xi, w) + b
                yh = int(z >= 0)
                d  = self.lr * (yi - yh)
                w += d * xi
                b += d
                e += int(d != 0)
            self.errors.append(e)
        self.weights = w
        self.bias    = b
        return self

    def predict(self, X):
        return (X @ self.weights + self.bias >= 0).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


X_train = np.load('outputs/X_train.npy')
X_test  = np.load('outputs/X_test.npy')
y_train = np.load('outputs/y_train.npy')
y_test  = np.load('outputs/y_test.npy')

lrs_gd   = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
lrs_perc = [0.001, 0.01,  0.05, 0.1,  0.3, 0.5]

gd_results   = {}
perc_results = {}

print("Gradient Descent — Learning Rate Sweep")
print(f"{'LR':>8} | {'Train':>8} | {'Test':>8} | {'Final Loss':>12}")
print("-" * 44)
for lr in lrs_gd:
    m = GDModel(lr=lr, n_epochs=100)
    m.fit(X_train, y_train)
    tr = m.score(X_train, y_train)
    te = m.score(X_test,  y_test)
    gd_results[lr] = {'train': tr, 'test': te, 'loss': m.loss, 'acc': m.acc}
    print(f"{lr:>8.3f} | {tr*100:>7.2f}% | {te*100:>7.2f}% | {m.loss[-1]:>12.6f}")

print("\nPerceptron — Learning Rate Sweep")
print(f"{'LR':>8} | {'Train':>8} | {'Test':>8} | {'Final Errors':>14}")
print("-" * 46)
for lr in lrs_perc:
    m = PerceptronModel(lr=lr, n_epochs=50)
    m.fit(X_train, y_train)
    tr = m.score(X_train, y_train)
    te = m.score(X_test,  y_test)
    perc_results[lr] = {'train': tr, 'test': te, 'errors': m.errors}
    print(f"{lr:>8.3f} | {tr*100:>7.2f}% | {te*100:>7.2f}% | {m.errors[-1]:>14d}")

palette_gd   = plt.cm.plasma(np.linspace(0.1, 0.9, len(lrs_gd)))
palette_perc = plt.cm.viridis(np.linspace(0.1, 0.9, len(lrs_perc)))

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Learning Rate Tuning — Both Algorithms', fontsize=13, fontweight='bold')

ax = axes[0, 0]
for lr, col in zip(lrs_gd, palette_gd):
    ax.plot(gd_results[lr]['loss'], label=f'lr={lr}', linewidth=1.8, color=col)
ax.set_title('GD: Loss per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = axes[0, 1]
for lr, col in zip(lrs_gd, palette_gd):
    ax.plot(gd_results[lr]['acc'], label=f'lr={lr}', linewidth=1.8, color=col)
ax.set_title('GD: Train Accuracy per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax   = axes[0, 2]
x    = np.arange(len(lrs_gd))
w    = 0.35
tr_v = [gd_results[lr]['train'] * 100 for lr in lrs_gd]
te_v = [gd_results[lr]['test']  * 100 for lr in lrs_gd]
ax.bar(x - w/2, tr_v, w, label='Train', color='#3498db', edgecolor='k', lw=0.5)
ax.bar(x + w/2, te_v, w, label='Test',  color='#e74c3c', edgecolor='k', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([str(lr) for lr in lrs_gd], fontsize=8)
ax.set_ylim(0, 115)
ax.set_title('GD: Accuracy by Learning Rate')
ax.set_xlabel('Learning Rate')
ax.legend()
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 0]
for lr, col in zip(lrs_perc, palette_perc):
    ax.plot(perc_results[lr]['errors'], label=f'lr={lr}', linewidth=1.8, color=col)
ax.set_title('Perceptron: Errors per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Misclassifications')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax   = axes[1, 1]
x    = np.arange(len(lrs_perc))
tr_v = [perc_results[lr]['train'] * 100 for lr in lrs_perc]
te_v = [perc_results[lr]['test']  * 100 for lr in lrs_perc]
ax.bar(x - w/2, tr_v, w, label='Train', color='#2ecc71', edgecolor='k', lw=0.5)
ax.bar(x + w/2, te_v, w, label='Test',  color='#e67e22', edgecolor='k', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([str(lr) for lr in lrs_perc], fontsize=8)
ax.set_ylim(0, 115)
ax.set_title('Perceptron: Accuracy by Learning Rate')
ax.set_xlabel('Learning Rate')
ax.legend()
ax.grid(axis='y', alpha=0.3)

conv_epochs = []
for lr in lrs_gd:
    accs = gd_results[lr]['acc']
    try:
        ep = next(i for i, a in enumerate(accs) if a >= 0.95)
    except StopIteration:
        ep = 100
    conv_epochs.append(ep)

ax = axes[1, 2]
ax.plot([str(lr) for lr in lrs_gd], conv_epochs,
        marker='o', color='#9b59b6', linewidth=2, markersize=8)
ax.set_title('GD: Epochs to Reach 95% Train Accuracy')
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Epoch')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/portion5_lr_tuning.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to outputs/portion5_lr_tuning.png")
