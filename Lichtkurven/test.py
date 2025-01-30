import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameter
R_major = 3  # Haupt-Radius des Torus (Abstand zur Torusmitte)
R_minor = 1  # Kleiner Radius des Torus (Querschnitt)
mu_0 = 4 * np.pi * 1e-7  # Magnetische Permeabilität
I = 10  # Strom durch den Draht
N = 100  # Anzahl der Windungen (für die Spule)

# Meshgrid für Torus-Koordinaten
theta = np.linspace(0, 2 * np.pi, 50)  # Winkel um die Hauptachse (Torusring)
phi = np.linspace(0, 2 * np.pi, 50)    # Winkel im Querschnitt (kleiner Radius)
theta, phi = np.meshgrid(theta, phi)

# Torus-Koordinaten
x = (R_major + R_minor * np.cos(phi)) * np.cos(theta)
y = (R_major + R_minor * np.cos(phi)) * np.sin(theta)
z = R_minor * np.sin(phi)

# Magnetfeld im Inneren des Torus (vereinfachtes Modell)
B_theta = (mu_0 * N * I) / (2 * np.pi * (R_major + R_minor * np.cos(phi)))

# Magnetfeld-Komponenten in 3D
Bx = -B_theta * np.sin(theta)
By = B_theta * np.cos(theta)
Bz = np.zeros_like(Bx)  # In der einfachen Darstellung keine z-Komponente

# Reduzierung der Dichte für das Plotten der Vektoren
step = 4  # Schrittweite für reduzierte Darstellung
x_sparse = x[::step, ::step]
y_sparse = y[::step, ::step]
z_sparse = z[::step, ::step]
Bx_sparse = Bx[::step, ::step]
By_sparse = By[::step, ::step]
Bz_sparse = Bz[::step, ::step]

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Torusfläche (zur Orientierung)
ax.plot_surface(x, y, z, color='lightblue', alpha=0.3, edgecolor='k', linewidth=0.2)

# Magnetfeldlinien
ax.quiver(
    x_sparse, y_sparse, z_sparse, 
    Bx_sparse, By_sparse, Bz_sparse, 
    length=0.3, color='red', normalize=True, linewidth=0.5, alpha=0.7
)

# Achsen und Ansicht
ax.set_title("Magnetfeld eines stromdurchflossenen Torus", fontsize=14)
ax.set_xlabel("X-Achse")
ax.set_ylabel("Y-Achse")
ax.set_zlabel("Z-Achse")
ax.view_init(elev=30, azim=30)
plt.show()
