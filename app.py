import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Generator Definition (same as training)
# -------------------------------

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

# -------------------------------
# Load Generator
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
G.load_state_dict(torch.load("generator_epoch_40.pth", map_location=device))
G.eval()

# -------------------------------
# Streamlit UI
# -------------------------------

st.title("🖋️ Handwritten Digit Generator")
digit = st.number_input("Enter a digit (0–9)", min_value=0, max_value=9, step=1)

if st.button("Generate Images"):
    z = torch.randn(5, 100).to(device)
    labels = torch.tensor([digit] * 5, dtype=torch.long).to(device)
    with torch.no_grad():
        images = G(z, labels).cpu()

    # Plot the images
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(images[i].squeeze(), cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
