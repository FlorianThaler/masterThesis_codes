# import numpy as np

# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

# x1 = np.linspace(-0.5, 0.5, 100)
# x2 = x1
# x3 = x1

# [xx1, xx2, xx3] = np.meshgrid(x1, x2, x3)

# zz = 10 * np.sin(50 * xx1 * xx2) + 20 * (xx3 - 0.25) ** 2

# [xx1, xx2] = np.meshgrid(x1, x2)

# zz = np.sin(50 * xx1 * xx2)

# fig = plt.figure()
# ax = fig.gca(projection = '3d')
# surf = ax.plot_surface(xx1, xx2, zz, antialiased = False, rstride = 1, cstride = 1, cmap = cm.coolwarm)
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')

# plt.show()


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# load some test data for demonstration and plot a wireframe
X, Y, Z = axes3d.get_test_data(0.1)
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)

# rotate the axes and update
for angle in range(0, 360):
	print(angle)
	ax.view_init(30, angle)
	plt.draw()
	plt.pause(.001)
