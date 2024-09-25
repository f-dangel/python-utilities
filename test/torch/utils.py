"""Utilities for the PyTorch tests."""

# the available PyTorch devices
from torch import cuda, device

DEVICES = [device("cpu"), device("cuda")] if cuda.is_available() else [device("cpu")]
DEVICE_IDS = [str(dev) for dev in DEVICES]
