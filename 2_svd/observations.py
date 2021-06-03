import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 3, 8, 30, 135]
k10 = [188, 145, 156, 242, 540, 1293]
k20 = [376, 289, 311, 484, 1080, 2585]
k30 = [563, 433, 466, 725, 1620, 3877]
k40 = [751, 577, 620, 967, 2159, 5169]
jpg = [60, 20 , 40, 120, 1500, 5040]

plt.plot(x, k10, color='#00e1ff', linewidth=2, label='k=10')
plt.plot(x, k20, color='#0099ff', linewidth=2, label='k=20')
plt.plot(x, k30, color='#004cff', linewidth=2, label='k=30')
plt.plot(x, k40, color='#1e00ff', linewidth=2, label='k=40')
plt.plot(x, jpg, color='#8c00ff', linewidth=2, label='jpg')

plt.xlabel("Image size (in 100000 pixels)")
plt.ylabel("Storage size (in kB)")
plt.legend()
plt.show()