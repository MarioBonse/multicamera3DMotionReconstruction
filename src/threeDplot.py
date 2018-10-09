"""
Utility funcitions for drowing
graphics
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

def update(num, data, line):
    line.set_data(data[:2, :num])
    line.set_3d_properties(data[2, :num])

"""
Function that animate points in 3d space
given x, y, z vectors
"""
def displayanimation(x, y, z, save):
    data = np.array([x, y, z])
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

    # Setting the axes properties
    ax.set_xlim3d([-0.2, 0.2])
    ax.set_xlabel('X')

    ax.set_ylim3d([-0.2, 0.2])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-0.2, 0.2])
    ax.set_zlabel('Z')

    ani = animation.FuncAnimation(fig, update, len(x), fargs=(data, line), interval=25, blit=False)
    if save:
        ani.save('matplot003.gif', writer='imagemagick')
    plt.show()
