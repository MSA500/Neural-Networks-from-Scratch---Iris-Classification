import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_d(a):
    return a * (1 - a)


class Perceptron:
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
                z = np.dot(xi, w) + b
                yh = int(z >= 0)

                d = self.lr * (yi - yh)

                w += d * xi
                b += d

                e += int(d != 0)

            self.errors.append(e)

        self.w = w
        self.b = b
        return self

    def predict(self, X):
        return (np.dot(X, self.w) + self.b >= 0).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class DeltaRule:
    def __init__(self, lr=0.1, n_epochs=100):
        self.lr = lr
        self.n_epochs = n_epochs
        self.loss = []

    def fit(self, X, y):
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0

        for _ in range(self.n_epochs):
            z = np.dot(X, w) + b
            out = sigmoid(z)
            err = y - out

            correction = err * sigmoid_d(out)

            w += self.lr * np.dot(X.T, correction) / n
            b += self.lr * correction.mean()

            mse = 0.5 * np.mean(err ** 2)
            self.loss.append(mse)

        self.w = w
        self.b = b
        return self

    def predict(self, X):
        out = sigmoid(np.dot(X, self.w) + self.b)
        return (out >= 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


def compute_metrics(model, X, y):
    yp = model.predict(X)

    TP = np.sum((yp == 1) & (y == 1))
    TN = np.sum((yp == 0) & (y == 0))
    FP = np.sum((yp == 1) & (y == 0))
    FN = np.sum((yp == 0) & (y == 1))

    acc = (TP + TN) / len(y)
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0

    return {
        'acc': acc,
        'rec': rec,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }


X_train = np.load('outputs/X_train.npy')
X_test = np.load('outputs/X_test.npy')
y_train = np.load('outputs/y_train.npy')
y_test = np.load('outputs/y_test.npy')


perc = Perceptron(lr=0.1, n_epochs=50)
perc.fit(X_train, y_train)

gd = DeltaRule(lr=0.1, n_epochs=100)
gd.fit(X_train, y_train)


pm_tr = compute_metrics(perc, X_train, y_train)
pm_te = compute_metrics(perc, X_test, y_test)

gm_tr = compute_metrics(gd, X_train, y_train)
gm_te = compute_metrics(gd, X_test, y_test)


print("Final Evaluation — Algorithm Comparison")
print(f"\n{'Metric':<14} {'Perc Train':>12} {'Perc Test':>12} {'GD Train':>12} {'GD Test':>12}")
print("-" * 64)

for key, label in [
    ('acc', 'Accuracy'),
    ('rec', 'Recall')
]:
    print(
        f"{label:<14} "
        f"{pm_tr[key] * 100:>11.2f}% "
        f"{pm_te[key] * 100:>11.2f}% "
        f"{gm_tr[key] * 100:>11.2f}% "
        f"{gm_te[key] * 100:>11.2f}%"
    )


print("\nPerceptron — Confusion Matrix (Test):")
print(f"  TN: {pm_te['TN']}   FP: {pm_te['FP']}")
print(f"  FN: {pm_te['FN']}   TP: {pm_te['TP']}")

print("\nDelta Rule — Confusion Matrix (Test):")
print(f"  TN: {gm_te['TN']}   FP: {gm_te['FP']}")
print(f"  FN: {gm_te['FN']}   TP: {gm_te['TP']}")


metric_labels = ['Accuracy', 'Recall']

perc_vals = [
    pm_te['acc'] * 100,
    pm_te['rec'] * 100
]

gd_vals = [
    gm_te['acc'] * 100,
    gm_te['rec'] * 100
]


fig = plt.figure(figsize=(12, 10))

fig.suptitle(
    'Final Algorithm Comparison — Perceptron vs Delta Rule',
    fontsize=14,
    fontweight='bold'
)

gs = gridspec.GridSpec(
    3,
    2,
    figure=fig,
    hspace=0.45,
    wspace=0.35
)


ax0 = fig.add_subplot(gs[0, :])

x = np.arange(len(metric_labels))
bar_width = 0.32

b1 = ax0.bar(
    x - bar_width / 2,
    perc_vals,
    bar_width,
    label='Perceptron',
    color='#3498db',
    edgecolor='k',
    linewidth=0.7
)

b2 = ax0.bar(
    x + bar_width / 2,
    gd_vals,
    bar_width,
    label='Delta Rule',
    color='#e74c3c',
    edgecolor='k',
    linewidth=0.7
)

ax0.set_xticks(x)
ax0.set_xticklabels(metric_labels)
ax0.set_ylim(0, 120)
ax0.set_ylabel('Score (%)')
ax0.set_title('Test Set Metrics — Side by Side')
ax0.legend()
ax0.grid(axis='y', alpha=0.3)

for bar, value in zip(b1, perc_vals):
    ax0.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.5,
        f'{value:.1f}%',
        ha='center',
        fontsize=8
    )

for bar, value in zip(b2, gd_vals):
    ax0.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.5,
        f'{value:.1f}%',
        ha='center',
        fontsize=8
    )


ax1 = fig.add_subplot(gs[1, 0])

ax1.plot(
    perc.errors,
    color='#3498db',
    linewidth=2,
    marker='o',
    markersize=3
)

ax1.set_title('Perceptron: Errors per Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Misclassifications')
ax1.grid(alpha=0.3)


ax2 = fig.add_subplot(gs[1, 1])

ax2.plot(
    gd.loss,
    color='#e74c3c',
    linewidth=2
)

ax2.set_title('Delta Rule: Loss per Epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE')
ax2.grid(alpha=0.3)


for idx, (m_te, title) in enumerate([
    (pm_te, 'Perceptron — Confusion Matrix'),
    (gm_te, 'Delta Rule — Confusion Matrix')
]):
    ax = fig.add_subplot(gs[2, idx])

    cm = np.array([
        [m_te['TN'], m_te['FP']],
        [m_te['FN'], m_te['TP']]
    ])

    ax.imshow(cm, cmap='Blues')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(['Pred 0', 'Pred 1'])
    ax.set_yticklabels(['Actual 0', 'Actual 1'])

    for r in range(2):
        for c in range(2):
            ax.text(
                c,
                r,
                str(cm[r, c]),
                ha='center',
                va='center',
                fontsize=14,
                fontweight='bold',
                color='white' if cm[r, c] > cm.max() / 2 else 'black'
            )

    ax.set_title(title, fontsize=9)


plt.savefig(
    'outputs/portion6_comparison.png',
    dpi=150,
    bbox_inches='tight'
)

print("\nPlot saved to outputs/portion6_comparison.png")
print("\nAll portions complete.")