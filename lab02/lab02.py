import numpy as np
import cv2
import matplotlib.pyplot as plt


from functions.functions import CalculateHistogramHSV, EqualizeImgHSV

alpha_val = 0.5
alpha_val2 = 1.5


img = cv2.imread("warm_picture.jpg").astype(np.float32)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(img)


new_img = cv2.merge((H, S, V))
histoH, histoS, histoV = CalculateHistogramHSV(new_img)

new_img = EqualizeImgHSV(new_img)
new_img = np.clip(new_img, 0, 255).astype(np.uint8)
new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)

cv2.imshow("HSV", new_img.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()




x = np.arange(256, dtype=int)
plt.figure()
plt.bar(x, histoV, align='center', color='blue' )
plt.show()



