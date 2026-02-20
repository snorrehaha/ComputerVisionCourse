from cgitb import grey

import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
import lpips as lpips


from enum import Enum

# Showing image via matplotlib, therfor RGB
class Color(Enum):
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)

Df = 0.5
k_values = [5, 15, 30]
s_values = [0, 1, 2, 4, 8]

canvas = np.zeros((512, 512, 3), np.uint8)
depth_map = np.zeros((512, 512), np.float32)

def drawCircle(canvas, depth_map, x, y, r, color, depth_val):
    cv2.circle(canvas, (x, y), r, color, -1)
    cv2.circle(depth_map, (x, y), r, depth_val, -1)





drawCircle(canvas,depth_map, 300, 300, 100, Color.RED.value, 0.2)
drawCircle(canvas,depth_map,200, 200, 75, Color.GREEN.value, 0.5)
drawCircle(canvas,depth_map,150, 150, 50, Color.BLUE.value, 1.0)

r = k_values[0] * np.abs(depth_map - Df)
r2 = k_values[1] * np.abs(depth_map - Df)
r3 = k_values[2] * np.abs(depth_map - Df)



blurred_images = [cv2.GaussianBlur(canvas, (0,0), sigma) for sigma in s_values]


# Plot
fig, axs = plt.subplots(2, 3, figsize=(12, 6))

canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
axs[0,0].imshow(canvas_rgb)

axs[0, 0].set_title("Canvas")
axs[0, 0].axis("off")

axs[0,1].imshow(depth_map, cmap="gray")
axs[0,1].set_title("Depth Map")
axs[0,1].axis("off")



r_max = np.max(r3)  # largest CoC across all k
axs[0,2].imshow(r, cmap="gray", vmin=0, vmax=r_max)
axs[0,2].set_title("COC")
axs[0,2].axis("off")

axs[1,0].imshow(r2, cmap="gray", vmin=0, vmax=r_max)
axs[1,0].set_title("COC")
axs[1,0].axis("off")

axs[1,1].imshow(r3, cmap="gray", vmin=0, vmax=r_max)
axs[1,1].set_title("COC")
axs[1,1].axis("off")

axs[1,2].axis("off")











plt.show()
