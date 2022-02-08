from matplotlib import pyplot as plt

import numpy as np

ls1 = [7.15, 14.37, 24.03]		# m = 5, r = 2, N = 10, 20, 30
ls2 = [39.58, 110.82, 177.88]	# m = 5, r = 6, N = 10, 20, 30
ls3 = [87.79, 295.06, 371.32]	# m = 5, r = 10, N = 10, 20, 30

ls4 = [27.01, 39.31, 39.78]		# m = 10, r = 2, N = 10, 20, 30
ls5 = [82.17, 155.32, 359.30]	# m = 10, r = 6, N = 10, 20, 30
ls6 = [189.73, 551.60, 762.51]	# m = 10, r = 10, N = 10, 20, 30

loc = np.arange(0, 3)

fig = plt.figure()

plt.subplot(121)

p1 = plt.bar(loc + 0.00, ls1, color = 'b', width = 0.25)
p2 = plt.bar(loc + 0.25, ls2, color = 'g', width = 0.25)
p3 = plt.bar(loc + 0.50, ls3, color = 'r', width = 0.25)

plt.xticks(np.arange(0, 3, step = 1) + 0.25, ('r = 2', 'r = 6', 'r = 10'))
plt.legend((p1[0], p2[0], p3[0]), ('N = 10', 'N = 20', 'N = 30'))

plt.ylabel('Time [s]')
plt.ylim([0, 800])
plt.title('CPU times (m = 5)')


plt.subplot(122)

p4 = plt.bar(loc + 0.00, ls4, color = 'b', width = 0.25)
p5 = plt.bar(loc + 0.25, ls5, color = 'g', width = 0.25)
p6 = plt.bar(loc + 0.50, ls6, color = 'r', width = 0.25)

plt.xticks(np.arange(0, 3, step = 1) + 0.25, ('r = 2', 'r = 6', 'r = 10'))
plt.legend((p4[0], p5[0], p6[0]), ('N = 10', 'N = 20', 'N = 30'))

plt.ylabel('Time [s]')
plt.ylim([0, 800])

plt.title('CPU times (m = 10)')

plt.subplots_adjust(wspace = 0.5)

plt.show()
