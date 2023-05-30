import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Your data
camera_distances = [
    [5, 2, 90],# [distance,z,fov]
    [10, 3, 90],
]

# Expand your data
xyzf_data = []
def expand_coordinates(d, z, fov):
    import math
    points = [
        [d, 0, z, fov], [-d, 0, z, fov], [0, d, z, fov], [0, -d, z, fov],
        [d * math.cos(math.radians(45)), d * math.sin(math.radians(45)), z, fov],
        [d * math.cos(math.radians(135)), d * math.sin(math.radians(135)), z, fov],
        [d * math.cos(math.radians(225)), d * math.sin(math.radians(225)), z, fov],
        [d * math.cos(math.radians(315)), d * math.sin(math.radians(315)), z, fov]
    ]
    return points

for [d, z, fov] in camera_distances:
    for p in expand_coordinates(d, z, fov):
        xyzf_data.append(p)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.grid(False)     # Cancel the grid

# Draw each point and the line connecting to the origin
for [x, y, z,fov] in xyzf_data:
    ax.scatter(x, y, z, c='r')                          # Point color
    ax.text(x, y, z, f'({x}, {y}, {z})')                # Coordinate
    ax.plot([0, x], [0, y], [0, z], c='r', linewidth=1) # Line color and thickness

# Connect the points in a circle on the plane
# for i in range(0, len(xyz_data), 4):
#     # compute circular points on the plane
#     center = [np.mean([xyz_data[i + j][k] for j in range(4)]) for k in range(3)]
#     radius = np.linalg.norm(np.array(center) - np.array(xyz_data[i][: 3]))
#     circle_points = [
#         center + radius * np.array([np.cos(theta), np.sin(theta), 0]) for theta in np.linspace(0, 2 * np.pi, 100)
#     ]
#     ax.plot(
#         [p[0] for p in circle_points], [p[1] for p in circle_points], [p[2] for p in circle_points], 'r--'
#     )                                                                                                      # dashed line

# Hide axes
ax.set_axis_off()

# Change your view angle
ax.view_init(elev=90, azim=0) # Change these numbers to your liking

# Save figures
fig.savefig("figure.png")
# fig.savefig("figure.pdf")
