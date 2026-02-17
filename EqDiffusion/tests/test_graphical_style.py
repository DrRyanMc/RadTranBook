#!/usr/bin/env python3
"""
Test the new graphical style for efficiency study plots
"""

import numpy as np
import matplotlib.pyplot as plt

def test_plot_style():
    """Demonstrate the new plotting style"""
    
    print("="*70)
    print("Testing New Graphical Style")
    print("="*70)
    
    # Create mock data
    dt_values = np.array([0.1, 0.05, 0.01, 0.005, 0.001])
    
    # Methods with new style
    methods = [
        {
            'name': 'Implicit Euler (one iter)',
            'color': 'blue',
            'marker': 'o',
            'linestyle': '--',
            'fillstyle': 'full'
        },
        {
            'name': 'Implicit Euler (converged)',
            'color': 'blue',
            'marker': 'o',
            'linestyle': '-',
            'fillstyle': 'none'
        },
        {
            'name': 'Crank-Nicolson (one iter)',
            'color': 'green',
            'marker': 's',
            'linestyle': '--',
            'fillstyle': 'full'
        },
        {
            'name': 'Crank-Nicolson (converged)',
            'color': 'green',
            'marker': 's',
            'linestyle': '-',
            'fillstyle': 'none'
        },
        {
            'name': 'TR-BDF2 (one iter)',
            'color': 'red',
            'marker': '^',
            'linestyle': '--',
            'fillstyle': 'full'
        },
        {
            'name': 'TR-BDF2 (converged)',
            'color': 'red',
            'marker': '^',
            'linestyle': '-',
            'fillstyle': 'none'
        }
    ]
    
    # Generate mock error data
    np.random.seed(42)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, method in enumerate(methods):
        # Create error data (lower for converged, higher for one iter)
        if 'converged' in method['name']:
            errors = 0.01 * dt_values**2 * (1 + 0.1*np.random.randn(len(dt_values)))
        else:
            errors = 0.05 * dt_values * (1 + 0.1*np.random.randn(len(dt_values)))
        
        ax.loglog(dt_values, errors,
                 marker=method['marker'],
                 color=method['color'],
                 linestyle=method['linestyle'],
                 fillstyle=method['fillstyle'],
                 label=method['name'],
                 markersize=10,
                 linewidth=2.5,
                 markeredgewidth=2)
    
    # Reference lines
    ax.loglog(dt_values, 0.01 * dt_values, 'k--', alpha=0.3, linewidth=1, label='O(Δt)')
    ax.loglog(dt_values, 0.01 * dt_values**2, 'k:', alpha=0.3, linewidth=1, label='O(Δt²)')
    
    ax.set_xlabel('Time step Δt (ns)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative L2 Error', fontsize=14, fontweight='bold')
    ax.set_title('New Graphical Style: Converged vs One Iteration', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Create legend with explanation
    legend = ax.legend(fontsize=11, loc='upper left', 
                      title='Solid line + open marker = converged\\nDashed line + filled marker = one iter',
                      title_fontsize=10)
    legend.get_title().set_multialignment('left')
    
    ax.grid(True, alpha=0.3, which='both')
    
    # Add text annotation
    ax.text(0.98, 0.02, 
            'Style Guide:\\n• Same color = same method\\n• Solid + open = converged\\n• Dashed + filled = one iter',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('test_graphical_style.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'test_graphical_style.png'")
    print("\nStyle convention:")
    print("  • Same color = same method")
    print("  • Solid line + open marker = converged")
    print("  • Dashed line + filled marker = one iteration")
    print("\n" + "="*70)
    # plt.show()


if __name__ == "__main__":
    test_plot_style()
