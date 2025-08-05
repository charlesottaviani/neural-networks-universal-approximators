import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Définir la fonction indicatrice : 1 si x > 0, 0 sinon
def target_function(x):
    return x*torch.sin(1/x)

# Données d'entraînement : x entre -1 et 1
x_train = torch.linspace(-1, 1, 500).unsqueeze(1)
y_train = target_function(x_train)

# Définition d'une sigmoïde raide paramétrée
class SharpSigmoid(nn.Module):
    def __init__(self, slope=1):
        super().__init__()
        self.slope = slope

    def forward(self, x):
        return 1 / (1 + torch.exp(-self.slope * x))

# Réseau de neurones à une couche cachée avec activation personnalisée
class OneHiddenLayerNet(nn.Module):
    def __init__(self, hidden_size=20, slope=1.0):
        super().__init__()
        self.hidden = nn.Linear(1, hidden_size)
        self.activation = SharpSigmoid(slope)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

# Liste des pentes à tester
slopes = [1, 2, 5, 10, 20]
results = {}

# Entraînement pour chaque valeur de k
for k in slopes:
    print(f"Entraînement pour k = {k}")
    model = OneHiddenLayerNet(hidden_size=20, slope=k)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(300):  # tu peux augmenter à 1000 si besoin
        model.train()
        output = model(x_train)
        loss = criterion(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Prédiction
    model.eval()
    x_test = torch.linspace(-1, 1, 500).unsqueeze(1)
    y_pred = model(x_test).detach()
    results[k] = y_pred

# Tracé des résultats
x_np = x_test.squeeze().numpy()
y_true = target_function(x_test).numpy()

plt.figure(figsize=(10, 6))
plt.plot(x_np, y_true, label="Fonction indicatrice", color='black', linewidth=2)

for k, y_pred in results.items():
    plt.plot(x_np, y_pred.numpy(), label=f"Approximation (k={k})")

plt.title("Approximation d'une fonction indicatrice avec différentes pentes de sigmoïde")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
