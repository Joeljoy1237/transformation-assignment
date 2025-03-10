import numpy as np
import matplotlib.pyplot as plt

def translation_matrix(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

def rotation_matrix(angle):
    theta = np.radians(angle)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def scaling_matrix(sx, sy):
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])

def transform_points(points, matrix):
    homogeneous_points = np.vstack((points, np.ones((1, points.shape[1]))))
    transformed_points = matrix @ homogeneous_points
    return transformed_points[:2, :]

points = np.array([
    [0, 0, 2, 2, 0],
    [0, 2, 2, 0, 0]
])

tx, ty = -3, 4
angle = 30
sx, sy = 2, 1

T = translation_matrix(tx, ty)
R = rotation_matrix(angle)
S = scaling_matrix(sx, sy)

composite_matrix = S @ R @ T
transformed_points = transform_points(points, composite_matrix)

plt.figure(figsize=(8, 8))
plt.plot(points[0, :], points[1, :], label="Original", marker='o')
plt.plot(transformed_points[0, :], transformed_points[1, :], label="Transformed", marker='x')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.title("Composite 2D Transformations on Square")
plt.show()
