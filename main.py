import numpy as np
import matplotlib.pyplot as plt

def solve_z(X, Y, a, b, c, d, e, f, g, h, i, j, epsilon=1e-8):
    """
    Solves for Z in the quadric equation:
        Q(x, y, z) = ax^2 + by^2 + cz^2 + dxy + exz + fyz + gx + hy + iz + j = 0.
    Handles both quadratic and linear cases for z, with tolerance for floating-point issues.
    """
    Z1 = np.full_like(X, np.nan)
    Z2 = np.full_like(X, np.nan)

    # Quadratic case: cz^2 + Bz + C = 0
    if c != 0:
        B = e * X + f * Y + i
        C = a * X**2 + b * Y**2 + d * X * Y + g * X + h * Y + j
        discriminant = B**2 - 4 * c * C

        # Allow a small tolerance to prevent floating-point gaps
        mask = discriminant >= -epsilon
        discriminant[~mask] = 0  # Set invalid discriminants to zero
        Z1[mask] = (-B[mask] + np.sqrt(np.maximum(discriminant[mask], 0))) / (2 * c)
        Z2[mask] = (-B[mask] - np.sqrt(np.maximum(discriminant[mask], 0))) / (2 * c)
    else:
        # Linear case: Bz + C = 0 -> z = -C / B
        B = e * X + f * Y + i
        C = a * X**2 + b * Y**2 + d * X * Y + g * X + h * Y + j

        mask = np.abs(B) > epsilon  # Avoid division by near-zero B
        Z1[mask] = -C[mask] / B[mask]
        Z2[mask] = Z1[mask]  # Single solution repeated

    return Z1, Z2

def plot_quadric(a, b, c, d, e, f, g, h, i, j, xlim=(-10, 10), ylim=(-10, 10), resolution=200):
    """
    Plots a 3D quadric surface based on the general form:
        Q(x, y, z) = ax^2 + by^2 + cz^2 + dxy + exz + fyz + gx + hy + iz + j = 0.
    """
    # Generate meshgrid for X and Y
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Solve for Z
    Z1, Z2 = solve_z(X, Y, a, b, c, d, e, f, g, h, i, j)

    # Plot surfaces
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot real solutions
    if np.isfinite(Z1).any():
        ax.plot_surface(X, Y, Z1, color='b', alpha=0.6, edgecolor='none', label="Surface 1")
    if np.isfinite(Z2).any():
        ax.plot_surface(X, Y, Z2, color='r', alpha=0.6, edgecolor='none', label="Surface 2")

    # Set axis labels and limits
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(xlim)

    plt.title("3D Quadric Surface")
    plt.show()

# User Input for Coefficients
print("Enter the coefficients for the quadric equation:")
a = float(input("Coefficient of x^2 (a): "))
b = float(input("Coefficient of y^2 (b): "))
c = float(input("Coefficient of z^2 (c): "))
d = float(input("Coefficient of xy (d): "))
e = float(input("Coefficient of xz (e): "))
f = float(input("Coefficient of yz (f): "))
g = float(input("Coefficient of x (g): "))
h = float(input("Coefficient of y (h): "))
i = float(input("Coefficient of z (i): "))
j = float(input("Constant term (j): "))

# Plot the surface
plot_quadric(a, b, c, d, e, f, g, h, i, j)
