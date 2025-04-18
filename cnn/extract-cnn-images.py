import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CDQN import CDQN  # Make sure this is in the same directory or adjust import
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the model
model = CDQN().to(device)
model.load_state_dict(torch.load("cnn/trained_model/cnn_dqn_model_20250404-234200.pth", map_location=device))
model.eval()

# 2. Plot Conv Filters
def plot_conv_filters(layer, title, channels=0):
    weights = layer.weight.data.cpu().numpy()
    num_filters = weights.shape[0]
    fig, axes = plt.subplots(1, num_filters, figsize=(num_filters * 2, 2))
    for i in range(num_filters):
        ax = axes[i]
        ax.imshow(weights[i][channels], cmap='gray')
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

print("🎨 Visualizing Filters...")
plot_conv_filters(model.conv1, "Conv1 Filters", channels=0)
plot_conv_filters(model.conv2, "Conv2 Filters", channels=0)

# 3. Visualize Feature Maps from Dummy Input
def visualize_feature_maps(model, input_tensor):
    # Forward pass until each layer
    x = F.relu(model.bn1(model.conv1(input_tensor)))
    fmap1 = x.detach().cpu().numpy()[0]  # shape [C, H, W]
    x = F.max_pool2d(x, kernel_size=2, stride=1)
    x = F.relu(model.bn2(model.conv2(x)))
    fmap2 = x.detach().cpu().numpy()[0]

    def plot_maps(fmap, title):
        num_maps = fmap.shape[0]
        fig, axes = plt.subplots(1, num_maps, figsize=(num_maps * 2, 2))
        for i in range(num_maps):
            ax = axes[i]
            ax.imshow(fmap[i], cmap='viridis')
            ax.axis('off')
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    print("📊 Feature Maps from Conv1:")
    plot_maps(fmap1, "Feature Maps after Conv1")

    print("📊 Feature Maps from Conv2:")
    plot_maps(fmap2, "Feature Maps after Conv2")

# 4. Generate dummy input (3 channels, 8x8 grid)
dummy_input = torch.rand(1, 3, 8, 8).to(device)
visualize_feature_maps(model, dummy_input)
