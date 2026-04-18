# Central configuration — all magic numbers and domain constants live here.

# Model architecture
ENCODER_NAME = 'resnet18'
NUM_CLASSES = 3
IN_CHANNELS = 1  # grayscale

# Image dimensions used for training and inference
TARGET_H = 256
TARGET_W = 320

# Segmentation class definitions
CLASS_NAMES = ['Background', 'Edge Artifacts', 'Defects']
CLASS_COLORS = {0: [0, 0, 0], 1: [0, 255, 0], 2: [255, 0, 0]}  # BGR: black, green, red

# Dataset samples
SAMPLES = [
    'A-09_1_layer', 'A-09_2_layer', 'A-09_3_layer',
    'T-16_1_layer', 'T-16_2_layer', 'T-16_3_layer',
    'X5Y4_1_layer', 'X5Y4_2_layer', 'X5Y4_3_layer',
]
