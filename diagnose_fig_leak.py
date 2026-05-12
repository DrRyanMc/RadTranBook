#!/usr/bin/env python3
"""
Diagnose memory growth after repeated figure saves.
Simulates exactly what plot_solution does on the 168x354 mesh.
"""
import tracemalloc, gc, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers

tracemalloc.start(25)

sys.path.insert(0, 'nonEquilibriumDiffusion')
sys.path.insert(0, 'utils')
from plotfuncs import show


def mem_rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def fake_plot_solution(step_n):
    """Mirrors plot_solution exactly: pcolormesh on 168x354 + colorbar + save."""
    data = np.random.rand(168, 354)

    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 3 * 1.275))
    im1 = ax1.pcolormesh(data, shading='auto', cmap='plasma', vmin=0, vmax=1)
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal',
                         location='top', pad=0.15, label='T (keV)')
    ax1.set_aspect('equal')
    plt.tight_layout()
    show(f'/tmp/test_mat_{step_n}.png', close_after=True, cbar_ax=cbar1.ax)

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 3 * 1.275))
    im2 = ax2.pcolormesh(data, shading='auto', cmap='plasma', vmin=0, vmax=1)
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal',
                         location='top', pad=0.15, label='Tr (keV)')
    ax2.set_aspect('equal')
    plt.tight_layout()
    show(f'/tmp/test_rad_{step_n}.png', close_after=True, cbar_ax=cbar2.ax)


print(f'Before any figures: {mem_rss_mb():.1f} MB RSS')
snap_pre = tracemalloc.take_snapshot()

fake_plot_solution(1)
gc.collect()
n_open = len(matplotlib._pylab_helpers.Gcf.get_all_fig_managers())
print(f'After save 1 (post GC): {mem_rss_mb():.1f} MB RSS  open figs={n_open}')
snap1 = tracemalloc.take_snapshot()

fake_plot_solution(2)
gc.collect()
n_open = len(matplotlib._pylab_helpers.Gcf.get_all_fig_managers())
print(f'After save 2 (post GC): {mem_rss_mb():.1f} MB RSS  open figs={n_open}')
snap2 = tracemalloc.take_snapshot()

fake_plot_solution(3)
gc.collect()
n_open = len(matplotlib._pylab_helpers.Gcf.get_all_fig_managers())
print(f'After save 3 (post GC): {mem_rss_mb():.1f} MB RSS  open figs={n_open}')
snap3 = tracemalloc.take_snapshot()

fake_plot_solution(4)
gc.collect()
n_open = len(matplotlib._pylab_helpers.Gcf.get_all_fig_managers())
print(f'After save 4 (post GC): {mem_rss_mb():.1f} MB RSS  open figs={n_open}')
snap4 = tracemalloc.take_snapshot()

print()
print('=== Top tracemalloc growth snap1 -> snap2 ===')
for s in snap2.compare_to(snap1, 'lineno')[:15]:
    if abs(s.size_diff) > 500:
        print(s)

print()
print('=== Top tracemalloc growth snap2 -> snap3 ===')
for s in snap3.compare_to(snap2, 'lineno')[:15]:
    if abs(s.size_diff) > 500:
        print(s)

print()
print('=== Top tracemalloc growth snap3 -> snap4 ===')
for s in snap4.compare_to(snap3, 'lineno')[:15]:
    if abs(s.size_diff) > 500:
        print(s)

print()
print('=== Any figures still alive after all closes ===')
for fm in matplotlib._pylab_helpers.Gcf.get_all_fig_managers():
    print(' ', fm)
