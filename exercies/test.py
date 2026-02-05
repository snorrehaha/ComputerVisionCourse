import numpy as np
import matplotlib.pyplot as plt

# CIE 1931 2° Standard Observer color matching functions
# Wavelength range: 380-780 nm in 5nm steps
wavelengths = np.arange(380, 785, 5)

# CIE 1931 2° Standard Observer CMFs (simplified dataset)
# These are the x̄(λ), ȳ(λ), z̄(λ) functions
cmf_x = np.array([
    0.0014, 0.0022, 0.0042, 0.0076, 0.0143, 0.0232, 0.0435, 0.0776, 0.1344, 0.2148,
    0.2839, 0.3285, 0.3483, 0.3481, 0.3362, 0.3187, 0.2908, 0.2511, 0.1954, 0.1421,
    0.0956, 0.0580, 0.0320, 0.0147, 0.0049, 0.0024, 0.0093, 0.0291, 0.0633, 0.1096,
    0.1655, 0.2257, 0.2904, 0.3597, 0.4334, 0.5121, 0.5945, 0.6784, 0.7621, 0.8425,
    0.9163, 0.9786, 1.0263, 1.0567, 1.0622, 1.0456, 1.0026, 0.9384, 0.8544, 0.7514,
    0.6424, 0.5419, 0.4479, 0.3608, 0.2835, 0.2187, 0.1649, 0.1212, 0.0874, 0.0636,
    0.0468, 0.0329, 0.0227, 0.0158, 0.0114, 0.0081, 0.0058, 0.0041, 0.0029, 0.0020,
    0.0014, 0.0010, 0.0007, 0.0005, 0.0003, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0000
])

cmf_y = np.array([
    0.0000, 0.0001, 0.0001, 0.0002, 0.0004, 0.0006, 0.0012, 0.0022, 0.0040, 0.0073,
    0.0116, 0.0168, 0.0230, 0.0298, 0.0380, 0.0480, 0.0600, 0.0739, 0.0910, 0.1126,
    0.1390, 0.1693, 0.2080, 0.2586, 0.3230, 0.4073, 0.5030, 0.6082, 0.7100, 0.7932,
    0.8620, 0.9149, 0.9540, 0.9803, 0.9950, 1.0000, 0.9950, 0.9786, 0.9520, 0.9154,
    0.8700, 0.8163, 0.7570, 0.6949, 0.6310, 0.5668, 0.5030, 0.4412, 0.3810, 0.3210,
    0.2650, 0.2170, 0.1750, 0.1382, 0.1070, 0.0816, 0.0610, 0.0446, 0.0320, 0.0232,
    0.0170, 0.0119, 0.0082, 0.0057, 0.0041, 0.0029, 0.0021, 0.0015, 0.0010, 0.0007,
    0.0005, 0.0004, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000
])

cmf_z = np.array([
    0.0065, 0.0105, 0.0201, 0.0362, 0.0679, 0.1102, 0.2074, 0.3713, 0.6456, 1.0391,
    1.3856, 1.6230, 1.7471, 1.7826, 1.7721, 1.7441, 1.6692, 1.5281, 1.2876, 1.0419,
    0.8130, 0.6162, 0.4652, 0.3533, 0.2720, 0.2123, 0.1582, 0.1117, 0.0782, 0.0573,
    0.0422, 0.0298, 0.0203, 0.0134, 0.0087, 0.0057, 0.0039, 0.0027, 0.0021, 0.0018,
    0.0017, 0.0014, 0.0011, 0.0010, 0.0008, 0.0006, 0.0003, 0.0002, 0.0002, 0.0001,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
])

# CIE D65 Standard Illuminant spectral power distribution
# Relative spectral power distribution
d65_spd = np.array([
    49.98, 52.31, 54.65, 68.70, 82.75, 87.12, 91.49, 92.46, 93.43, 90.06,
    86.68, 95.77, 104.86, 110.94, 117.01, 117.41, 117.81, 116.34, 114.86, 115.39,
    115.92, 112.37, 108.81, 109.08, 109.35, 108.58, 107.80, 106.30, 104.79, 106.24,
    107.69, 106.05, 104.41, 104.23, 104.05, 102.02, 100.00, 98.17, 96.33, 96.06,
    95.79, 92.24, 88.69, 89.35, 90.01, 89.80, 89.60, 88.65, 87.70, 85.49,
    83.29, 83.49, 83.70, 81.86, 80.03, 80.12, 80.21, 81.25, 82.28, 80.28,
    78.28, 74.00, 69.72, 70.67, 71.61, 72.98, 74.35, 67.98, 61.60, 65.74,
    69.89, 72.49, 75.09, 69.34, 63.59, 55.01, 46.42, 56.61, 66.81, 65.09, 63.38
])


def calculate_xyz_tristimulus(reflectance, illuminant_spd, cmf_x, cmf_y, cmf_z, wavelengths):
    """
    Calculate CIE XYZ tristimulus values for a surface under given illuminant.

    Parameters:
    -----------
    reflectance : array-like
        Spectral reflectance of the surface (0-1)
    illuminant_spd : array-like
        Spectral power distribution of the illuminant
    cmf_x, cmf_y, cmf_z : array-like
        CIE color matching functions
    wavelengths : array-like
        Wavelength values

    Returns:
    --------
    X, Y, Z : float
        CIE XYZ tristimulus values
    """
    # Wavelength interval (assuming uniform spacing)
    delta_lambda = wavelengths[1] - wavelengths[0]

    # Calculate the normalizing constant k
    # k = 100 / Σ(S(λ) * ȳ(λ) * Δλ)
    k = 100.0 / np.sum(illuminant_spd * cmf_y * delta_lambda)

    # Calculate tristimulus values
    # X = k * Σ(S(λ) * R(λ) * x̄(λ) * Δλ)
    X = k * np.sum(illuminant_spd * reflectance * cmf_x * delta_lambda)
    Y = k * np.sum(illuminant_spd * reflectance * cmf_y * delta_lambda)
    Z = k * np.sum(illuminant_spd * reflectance * cmf_z * delta_lambda)

    return X, Y, Z


# For a perfect white Lambertian surface, reflectance = 1.0 at all wavelengths
white_lambertian_reflectance = np.ones_like(wavelengths)

# Calculate XYZ values
X, Y, Z = calculate_xyz_tristimulus(
    white_lambertian_reflectance,
    d65_spd,
    cmf_x,
    cmf_y,
    cmf_z,
    wavelengths
)

print("CIE XYZ Tristimulus Values for White Lambertian Surface under D65 Illuminant")
print("=" * 75)
print(f"X = {X:.4f}")
print(f"Y = {Y:.4f}")
print(f"Z = {Z:.4f}")
print("\nNormalized chromaticity coordinates:")
sum_xyz = X + Y + Z
x_chrom = X / sum_xyz
y_chrom = Y / sum_xyz
print(f"x = {x_chrom:.4f}")
print(f"y = {y_chrom:.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: D65 Illuminant SPD
axes[0, 0].plot(wavelengths, d65_spd, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Wavelength (nm)')
axes[0, 0].set_ylabel('Relative Power')
axes[0, 0].set_title('CIE D65 Illuminant Spectral Power Distribution')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Color Matching Functions
axes[0, 1].plot(wavelengths, cmf_x, 'r-', label='x̄(λ)', linewidth=2)
axes[0, 1].plot(wavelengths, cmf_y, 'g-', label='ȳ(λ)', linewidth=2)
axes[0, 1].plot(wavelengths, cmf_z, 'b-', label='z̄(λ)', linewidth=2)
axes[0, 1].set_xlabel('Wavelength (nm)')
axes[0, 1].set_ylabel('Tristimulus Value')
axes[0, 1].set_title('CIE 1931 2° Standard Observer Color Matching Functions')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: White Lambertian Reflectance
axes[1, 0].plot(wavelengths, white_lambertian_reflectance, 'k-', linewidth=2)
axes[1, 0].set_xlabel('Wavelength (nm)')
axes[1, 0].set_ylabel('Reflectance')
axes[1, 0].set_title('White Lambertian Surface Reflectance')
axes[1, 0].set_ylim([0, 1.2])
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: XYZ Bar Chart
xyz_values = [X, Y, Z]
xyz_labels = ['X', 'Y', 'Z']
colors = ['red', 'green', 'blue']
axes[1, 1].bar(xyz_labels, xyz_values, color=colors, alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('Tristimulus Value')
axes[1, 1].set_title('CIE XYZ Tristimulus Values')
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(xyz_values):
    axes[1, 1].text(i, v + 2, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()