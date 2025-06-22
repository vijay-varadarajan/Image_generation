import torch
import matplotlib.pyplot as plt
import torch.nn as nn

# Ensure Generator class matches training
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(10, 10)
        self.net = nn.Sequential(
            nn.Linear(110, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_embed(labels)
        x = torch.cat([z, c], dim=1)
        return self.net(x).view(-1, 1, 28, 28)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
G.load_state_dict(torch.load("generator_epoch_40.pth", map_location=device))
G.eval()

digit = 3
z = torch.randn(5, 100).to(device)
labels = torch.tensor([digit] * 5, dtype=torch.long).to(device)

with torch.no_grad():
    gen_images = G(z, labels).cpu()

# Plot the images
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(gen_images[i].squeeze(), cmap="gray")
    plt.axis("off")
plt.suptitle(f"Generated Images for Digit: {digit}")
plt.show()