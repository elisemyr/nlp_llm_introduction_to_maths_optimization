import matplotlib.pyplot as plt
import numpy as np
from exercice1 import f1
# Definition of the plotting domain
xmin, xmax, nx = -4, 4, 41
ymin, ymax, ny = -4, 4, 41
# Discretization of the plotting domain
x1d = np.linspace(xmin,xmax,nx)
y1d = np.linspace(ymin,ymax,ny)
x2d, y2d = np.meshgrid(x1d, y1d)

# Plot contour lines (isovalues) of f1
nIso = 21
plt.contour(x2d,y2d,f1(x2d,y2d),nIso)
plt.title('Contour Lines')
plt.xlabel('x values')
plt.ylabel('y values')
plt.grid()
plt.axis('square')