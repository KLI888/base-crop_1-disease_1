import os
import cv2
import numpy as np

# Create directories
os.makedirs('data/yes', exist_ok=True)
os.makedirs('data/no', exist_ok=True)

print("Generating dummy MRI image data...")

# Generate "Yes" tumor images (Contains a white circle)
for i in range(25):
    img_yes = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add some random noise
    img_yes += np.random.randint(0, 30, (100, 100, 3), dtype=np.uint8)
    # Draw a "tumor"
    cx, cy = np.random.randint(30, 70), np.random.randint(30, 70)
    radius = np.random.randint(10, 25)
    cv2.circle(img_yes, (cx, cy), radius, (200, 200, 200), -1)
    
    cv2.imwrite(f'data/yes/dummy_{i}.jpg', img_yes)
    
# Generate "No" tumor images (Just random noise/background)
for i in range(25):
    img_no = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add random noise
    img_no += np.random.randint(0, 40, (100, 100, 3), dtype=np.uint8)
    
    cv2.imwrite(f'data/no/dummy_{i}.jpg', img_no)

print("Saved 50 dummy images into data/yes and data/no!")
