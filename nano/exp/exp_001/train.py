"""
nano/train.py — Микро-обучение на tinygrad: двухслойный MLP учит функцию XOR.

Архитектура:
  Input(2) → Linear(16) + ReLU → Linear(1) + Sigmoid
  Loss: Binary Cross-Entropy
  Optimizer: SGD (lr=1.0)

Почему XOR:
  - Минимальная нелинейно-разделимая задача (линейная модель НЕ решит)
  - 4 примера, но требует скрытый слой → идеальный smoke-test для фреймворка
"""

from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters

# --- Модель ---
class TinyMLP:
    def __init__(self):
        self.l1 = Linear(2, 16)
        self.l2 = Linear(16, 1)

    def __call__(self, x: Tensor) -> Tensor:
        return self.l2(self.l1(x).relu()).sigmoid()

    def parameters(self):
        return get_parameters(self)

# --- Данные (XOR) ---
X = Tensor([[0,0],[0,1],[1,0],[1,1]], dtype=dtypes.float32)
Y = Tensor([[0],[1],[1],[0]], dtype=dtypes.float32)

# --- Обучение ---
Tensor.training = True
model = TinyMLP()
opt = SGD(model.parameters(), lr=1.0)

print("=== Обучение XOR на tinygrad ===\n")
for step in range(500):
    pred = model(X)
    # BCE loss: -[y*log(p) + (1-y)*log(1-p)]
    loss = -(Y * pred.log() + (1 - Y) * (1 - pred).log()).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 100 == 0 or step == 499:
        print(f"  step {step:3d}  loss={loss.item():.4f}")

# --- Инференс ---
print("\n=== Инференс ===\n")
preds = model(X)
for i in range(4):
    x0, x1 = int(X[i][0].item()), int(X[i][1].item())
    p = preds[i][0].item()
    print(f"  {x0} XOR {x1} = {p:.4f}  (ожидание: {int(Y[i][0].item())})")

print("\nГотово!")
