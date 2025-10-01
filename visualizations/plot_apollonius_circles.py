#!/usr/bin/env python3
"""
Generate publication-quality Apollonius circle visualization for three defenders
Creates filled circles showing defender dominance regions with proper mathematical notation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys

# Add parent directory to path for apollonius_solver import
sys.path.insert(0, '..')

def plot_three_defender_apollonius(attacker_pos, defender_positions, defender_speeds, attacker_speed):
    """
    Create publication-quality visualization of Apollonius circles for 3 defenders

    Args:
        attacker_pos: [x, y] position of attacker
        defender_positions: List of 3 [x, y] defender positions
        defender_speeds: List of 3 defender speeds
        attacker_speed: Attacker speed
    """

    # Import apollonius solver functions
    try:
        from apollonius_solver import solve_apollonius_optimization, apo_center, apo_radius
    except ImportError:
        print("Error: apollonius_solver.py not found. Please ensure it's in the parent directory.")
        return

    # Compute Apollonius circles for each defender
    circles = []
    for i, (def_pos, def_speed) in enumerate(zip(defender_positions, defender_speeds)):
        nu = def_speed / attacker_speed  # Speed ratio
        center = apo_center(def_pos, attacker_pos, nu)
        radius = apo_radius(def_pos, attacker_pos, nu)
        circles.append({
            'center': center,
            'radius': radius,
            'defender_idx': i
        })

    # Solve optimization to find optimal point
    result = solve_apollonius_optimization(
        attacker_pos=attacker_pos,
        defender_positions=defender_positions,
        nu=defender_speeds[0] / attacker_speed  # Assuming homogeneous for now
    )

    if result['success']:
        optimal_point = [result.get('min_x_coordinate', attacker_pos[0]), result['min_y_coordinate']]
    else:
        optimal_point = None

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set limits and aspect
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Add grid
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Plot boundaries
    ax.axhline(y=0, color='green', linewidth=2.5, zorder=1)  # Target line
    ax.axhline(y=1, color='red', linewidth=2.5, zorder=1)    # Spawn line

    # Colors for the three Apollonius circles (matching the image)
    circle_colors = ['#9999FF', '#99DD99', '#FFE4B5']  # Blue, Green, Orange/Beige
    circle_alphas = [0.4, 0.35, 0.3]  # Transparency levels

    # Sort circles by radius (largest first) for proper layering
    sorted_circles = sorted(circles, key=lambda c: c['radius'], reverse=True)

    # Draw filled Apollonius circles
    for circle, color, alpha in zip(sorted_circles, circle_colors, circle_alphas):
        circle_patch = Circle(
            circle['center'],
            circle['radius'],
            facecolor=color,
            edgecolor='black',
            linewidth=1.5,
            alpha=alpha,
            zorder=2
        )
        ax.add_patch(circle_patch)

    # Plot circle centers with labels C_1, C_2, C_3
    center_colors = ['green', 'blue', 'orange']
    for i, circle in enumerate(circles):
        ax.scatter(circle['center'][0], circle['center'][1],
                  color=center_colors[i], s=100, marker='o',
                  edgecolor='black', linewidth=1.5, zorder=5)
        # Add label near center
        ax.text(circle['center'][0] + 0.03, circle['center'][1] + 0.03,
               f'$C_{i+1}$', fontsize=14, fontweight='bold', zorder=6)

    # Plot attacker
    ax.scatter(attacker_pos[0], attacker_pos[1],
              color='red', s=200, marker='^',
              edgecolor='black', linewidth=2, zorder=6)
    ax.text(attacker_pos[0] + 0.02, attacker_pos[1] + 0.03,
           '$A$', fontsize=14, fontweight='bold', zorder=6)

    # Plot defenders
    for i, def_pos in enumerate(defender_positions):
        ax.scatter(def_pos[0], def_pos[1],
                  color='blue', s=150, marker='^',
                  edgecolor='black', linewidth=2, zorder=6)
        # Position label to avoid overlap
        offset_x = -0.05 if i == 0 else 0.02
        offset_y = -0.03 if i == 1 else 0.02
        ax.text(def_pos[0] + offset_x, def_pos[1] + offset_y,
               f'$D_{i+1}$', fontsize=14, fontweight='bold', zorder=6)

    # Plot optimal point if found
    if optimal_point:
        ax.scatter(optimal_point[0], optimal_point[1],
                  color='red', s=250, marker='*',
                  edgecolor='black', linewidth=2, zorder=7)
        # Add horizontal dashed line through optimal point
        ax.axhline(y=optimal_point[1], color='red', linestyle='--',
                  linewidth=1.5, alpha=0.7, zorder=3)

    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=circle_colors[0], edgecolor='black', alpha=circle_alphas[0], label='$A_1$'),
        Patch(facecolor=circle_colors[1], edgecolor='black', alpha=circle_alphas[1], label='$A_2$'),
        Patch(facecolor=circle_colors[2], edgecolor='black', alpha=circle_alphas[2], label='$A_3$'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
                  markersize=10, markeredgecolor='black', markeredgewidth=1.5, label='Attacker'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                  markersize=12, markeredgecolor='black', markeredgewidth=1.5,
                  label=f'Optimal ({optimal_point[0]:.2f}, {optimal_point[1]:.2f})' if optimal_point else 'Optimal')
    ]

    ax.legend(handles=legend_elements, loc='lower right', fontsize=11,
             framealpha=0.9, edgecolor='black')

    # Labels
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)

    # Set tick fontsize
    ax.tick_params(labelsize=12)

    plt.tight_layout()

    return fig, ax, optimal_point


def main():
    """Generate the three defender Apollonius visualization"""

    # Example configuration matching the image
    attacker_pos = np.array([0.5, 0.5])
    defender_positions = [
        np.array([0.2, 0.2]),   # D_1
        np.array([0.35, 0.1]),  # D_2
        np.array([0.35, 0.25])  # D_3
    ]

    # Speeds (all defenders faster than attacker)
    defender_speeds = [0.05, 0.05, 0.05]  # Homogeneous defenders
    attacker_speed = 0.015

    print("Generating three defender Apollonius circle visualization...")
    print(f"Attacker position: {attacker_pos}")
    print(f"Defender positions: {defender_positions}")
    print(f"Speed ratio (defender/attacker): {defender_speeds[0]/attacker_speed:.2f}")

    fig, ax, optimal_point = plot_three_defender_apollonius(
        attacker_pos, defender_positions, defender_speeds, attacker_speed
    )

    if optimal_point:
        print(f"\nOptimal capture point: ({optimal_point[0]:.2f}, {optimal_point[1]:.2f})")

    # Save figure
    output_file = 'Figure_three_defender.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {output_file}")

    # Also save PDF version
    output_pdf = 'Figure_three_defender.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"PDF version saved: {output_pdf}")

    plt.show()


if __name__ == "__main__":
    main()
