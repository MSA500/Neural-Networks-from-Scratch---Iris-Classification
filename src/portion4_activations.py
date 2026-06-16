import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def tanh(z):
    return np.tanh(z)

def tanh_d(a):
    return 1 - a ** 2

def relu(z):
    return np.maximum(0, z)

def relu_d(a):
    return (a > 0).astype(float)

def linear(z):
    return z

def linear_d(a):
    return np.ones_like(a)


ACTIVATIONS = {
    'Tanh':    (tanh,    tanh_d),
    'ReLU':    (relu,    relu_d),
    'Linear':  (linear,  linear_d),
}

COLORS = {
    'Tanh':    "#000000",
    'ReLU':    "#ff0000",
    'Linear':  "#2600ff",
}


class SingleNeuron:
    def __init__(self, activation='Sigmoid', lr=0.01, n_epochs=200):
        self.activation_name = activation
        self.act, self.act_d = ACTIVATIONS[activation]
        self.lr = lr
        self.n_epochs = n_epochs
        self.loss_history = []
        self.acc_history  = []

    def fitOnline(self,X, y):
        n,d = X.shape
        self.weights = np.zeros(d)
        self.bias = 0.0

        for _ in range(self.n_epochs):
            totalLoss = 0.0
            pred = []

            for xi,yi in zip(X,y):
                z = np.dot(xi,self.weights) + self.bias
                out = self.act(z)
                err = yi - out
                dErr = -(err)
                totalLoss += 0.5 * (err**2)
                pred.append(1 if out >= 0.5 else 0)

                grad_w = xi * dErr * self.act_d(out)
                self.weights -= self.lr * grad_w 

                grad_b = dErr * self.act_d(out)
                self.bias -= self.lr * grad_b

            acc  = np.mean(np.array(pred) == y)
            mse = totalLoss / n
            self.loss_history.append(mse)
            self.acc_history.append(acc)
        return self


    def fitBatch(self, X, y):
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0

        for _ in range(self.n_epochs):
            z   = X @ w + b
            out = self.act(z)
            err = y - out
            dErr = -(err)

            grad_w = X.T @ (dErr * self.act_d(out)) / n
            grad_b = (dErr * self.act_d(out)).mean()

            w -= self.lr * grad_w
            b -= self.lr * grad_b

            mse  = 0.5 * np.mean(err ** 2)
            pred = (out >= 0.5).astype(int)
            acc  = np.mean(pred == y)

            self.loss_history.append(mse)
            self.acc_history.append(acc)

        self.weights = w
        self.bias    = b
        return self

    def predict(self, X):
        out = self.act(X @ self.weights + self.bias)
        return (out >= 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


X_train = np.load('outputs/X_train.npy')
X_test  = np.load('outputs/X_test.npy')
y_train = np.load('outputs/y_train.npy')
y_test  = np.load('outputs/y_test.npy')

results = {}

print("Activation Function Experiments")
print(f"{'Activation':<12} {'Train Acc':>10} {'Test Acc':>10} {'Final Loss':>12}")
print("-" * 48)

for name in ACTIVATIONS:
    m = SingleNeuron(activation=name, lr=0.01, n_epochs=100)
    m.fitMine(X_train, y_train)
    tr = m.score(X_train, y_train)
    te = m.score(X_test,  y_test)
    results[name] = {
        'train': tr,
        'test':  te,
        'loss':  m.loss_history,
        'acc':   m.acc_history,
    }
    print(f"{name:<12} {tr*100:>9.2f}% {te*100:>9.2f}% {m.loss_history[-1]:>12.5f}")

z_range = np.linspace(-5, 5, 300)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Activation Function Experiments', fontsize=13, fontweight='bold')

ax = axes[0, 0]
for name, (fn, _) in ACTIVATIONS.items():
    ax.plot(z_range, fn(z_range), label=name, linewidth=2, color=COLORS[name])
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.set_title('Activation Functions')
ax.set_xlabel('z')
ax.set_ylabel('f(z)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 1]
for name in ACTIVATIONS:
    ax.plot(results[name]['loss'], label=name, linewidth=1.8, color=COLORS[name])
ax.set_title('Loss (MSE) per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 2]
for name in ACTIVATIONS:
    ax.plot(results[name]['acc'], label=name, linewidth=1.8, color=COLORS[name])
ax.set_title('Train Accuracy per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

names      = list(ACTIVATIONS.keys())
train_vals = [results[n]['train'] * 100 for n in names]
test_vals  = [results[n]['test']  * 100 for n in names]
x = np.arange(len(names))
w = 0.35

ax = axes[1, 0]
b1 = ax.bar(x - w/2, train_vals, w, label='Train', color='#3498db', edgecolor='k', lw=0.6)
b2 = ax.bar(x + w/2, test_vals,  w, label='Test',  color='#e67e22', edgecolor='k', lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0, 115)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy by Activation')
ax.legend()
ax.grid(axis='y', alpha=0.3)
for b, v in zip(b1, train_vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.5, f'{v:.0f}', ha='center', fontsize=7)
for b, v in zip(b2, test_vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.5, f'{v:.0f}', ha='center', fontsize=7)

final_losses = [results[n]['loss'][-1] for n in names]
ax = axes[1, 1]
bars = ax.bar(names, final_losses,
              color=[COLORS[n] for n in names], edgecolor='k', lw=0.7)
for b, v in zip(bars, final_losses):
    ax.text(b.get_x() + b.get_width()/2, v + 0.0005,
            f'{v:.4f}', ha='center', fontsize=8)
ax.set_ylabel('Final MSE Loss')
ax.set_title('Final Loss by Activation')
ax.grid(axis='y', alpha=0.3)

axes[1, 2].axis('off')
table_data = [
    [n, f"{results[n]['train']*100:.1f}%",
        f"{results[n]['test']*100:.1f}%",
        f"{results[n]['loss'][-1]:.5f}"]
    for n in names
]
tbl = axes[1, 2].table(
    cellText=table_data,
    colLabels=['Activation', 'Train Acc', 'Test Acc', 'Final Loss'],
    cellLoc='center', loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.6)
axes[1, 2].set_title('Summary Table', fontsize=10, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('outputs/portion4_activations.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to outputs/portion4_activations.png")
