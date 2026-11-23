import os
import time
import samna
import torch
import torch.nn as nn

from sinabs.layers import IAFSqueeze, SumPool2d
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.dynapcnn_visualizer import DynapcnnVisualizer


# Paths
weights_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "gesture_speck_conv_weights.pt"
)

icons_folder_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "icons"
)


# Build SNN
B = 1

snn = nn.Sequential(
    # Core 0
    nn.Conv2d(2, 8, kernel_size=3, stride=2, padding=0, bias=False),  # 128 -> 63
    IAFSqueeze(B),
    SumPool2d(2, 2),  # 63 -> 31

    # Core 1
    nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0, bias=False),  # 31 -> 29
    IAFSqueeze(B),
    SumPool2d(2, 2),  # 29 -> 14

    # Core 2
    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0, bias=False),  # 14 -> 12
    IAFSqueeze(B),
    SumPool2d(2, 2),  # 12 -> 6

    # Core 3
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=False),  # 6 -> 4
    IAFSqueeze(B),  # (64, 4, 4)

    # FC-style convs
    nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=0, bias=False),  # -> (64, 1, 1)
    IAFSqueeze(B),

    nn.Conv2d(64, 128, kernel_size=1, bias=False),
    IAFSqueeze(B),

    nn.Conv2d(128, 256, kernel_size=1, bias=False),
    IAFSqueeze(B),

    nn.Conv2d(256, 11, kernel_size=1, bias=False),  # -> (11, 1, 1)
    IAFSqueeze(B),
)

print(snn)


# Load weights
#if not os.path.exists(weights_path):
#    raise FileNotFoundError(f"Could not find weights file: {weights_path}")

state_dict = torch.load(weights_path, map_location="cpu")

snn.load_state_dict(state_dict, strict=True) #Strict=true should fix the problem
snn.eval()

print(f"\nLoaded weights from: {weights_path}")


# Deploy to Speck
chip_name = "speck2fdevkit"

dysnn = DynapcnnNetwork(
    snn,
    input_shape=(2, 128, 128),
    dvs_input=True,
    discretize=True,
)

# Monitor both the DVS input and the final output layer
dysnn.to(
    device=chip_name,
    monitor_layers=["dvs", -1],
    chip_layers_ordering="auto",
)

print("\nDeployed network:")
print(dysnn)


# Set up visualizer
readout_images = []
if os.path.isdir(icons_folder_path):
    readout_images = sorted(
        [
            os.path.join(icons_folder_path, f)
            for f in os.listdir(icons_folder_path)
            if not f.startswith(".")
        ]
    )

viz = DynapcnnVisualizer(
    window_scale=(4, 8),
    dvs_shape=(128, 128),
    add_power_monitor_plot=False,
    add_readout_plot=True,
    spike_collection_interval=500,
    readout_images=readout_images,
)

viz.connect(dysnn)


# Prevent automatic closing
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopping visualizer...")