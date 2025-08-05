import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1. Choix de la fonction cible
# -----------------------------
def target_function(x):
    return x*torch.sin(1/x)

# -----------------------------
# 2. Données d’apprentissage
# -----------------------------
x_train = torch.linspace(-1, 1, 200).unsqueeze(1)
y_train = target_function(x_train)

# -----------------------------
# 3. Définition du réseau de neurones
# -----------------------------

nb_neurones=1000
fonction_activation=nn.Sigmoid()

class OneHiddenLayerNet(nn.Module):
    def __init__(self, hidden_size=nb_neurones):
        super().__init__()
        self.hidden = nn.Linear(1, hidden_size)
        self.activation = fonction_activation
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

model = OneHiddenLayerNet(hidden_size=nb_neurones)

# -----------------------------
# 4. Entraînement du réseau
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1000 # Nombre d'entrainements
for epoch in range(epochs):
    model.train()
    output = model(x_train)
    loss = criterion(output, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

# -----------------------------
# 5. Évaluation & visualisation
# -----------------------------
model.eval()
x_test = torch.linspace(-1, 1, 200).unsqueeze(1)
y_pred = model(x_test).detach()
y_true = target_function(x_test)

plt.figure(figsize=(8, 5))
plt.plot(x_test.numpy(), y_true.numpy(), label="Fonction cible", linewidth=2)
plt.plot(x_test.numpy(), y_pred.numpy(), label="Approximation NN", linestyle='--')
plt.plot(x_test, torch.abs(y_pred - y_true), label="Ecart", linewidth=0.5)
plt.legend()
plt.title("Approximation d'une fonction continue par un réseau à 1 couche cachée")
plt.grid(True)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.tight_layout()
plt.show()
