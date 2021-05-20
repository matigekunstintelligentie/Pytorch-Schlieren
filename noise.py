import numpy as np
import cv2
import scipy.ndimage

# noise = 255*np.random.randint(0, 2, (int(1080*np.sqrt(2)), 1080)).astype('uint8')
noise = 255*np.random.choice([1,0], (int(1080*np.sqrt(2)), 1080), p=[0.8, 0.2]).astype('uint8')
# noise = 255*(scipy.ndimage.morphology.binary_dilation(noise, iterations=1).astype(noise.dtype))
print(noise)

cv2.imwrite( "noise.png", cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR));
