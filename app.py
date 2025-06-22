
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(110, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(-1, 1, 28, 28)

def create_label_vectors(labels, num_classes=10):
    label_vectors = torch.zeros((labels.size(0), num_classes))
    label_vectors[torch.arange(labels.size(0)), labels] = 1
    return label_vectors

st.title("Handwritten Digit Image Generator")

digit = st.selectbox("Choose a digit (0-9):", list(range(10)))

if st.button("Generate Images"):
    model = Generator()
    model.load_state_dict(torch.load("generator_model.pth", map_location=torch.device('cpu')))
    model.eval()

    noise = torch.randn(5, 100)
    labels = torch.tensor([digit] * 5)
    label_vectors = create_label_vectors(labels)
    input_vector = torch.cat((noise, label_vectors), dim=1)

    with torch.no_grad():
        images = model(input_vector)

    st.write(f"Generated images of digit {digit}:")
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(images[i][0], cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
