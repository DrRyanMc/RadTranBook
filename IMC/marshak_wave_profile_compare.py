import argparse
import json
import time

import numpy as np

import IMC1D as fc
import IMC1D_CarterForest as cf


def build_mesh(length, cells):
    mesh = np.zeros((cells, 2))
    dx = length / cells
    for i in range(cells):
        mesh[i] = [i * dx, (i + 1) * dx]
    return mesh


def summarize(method_name, module, mesh, t_init, tr_init, eos, inv_eos,
              sigma_a_f, cv, source, t_boundary, reflect, geometry,
              n_target, n_boundary, n_source, n_max, dt, t_final,
              extra_step_kwargs=None):
    extra_step_kwargs = extra_step_kwargs or {}
    np.random.seed(12345)
    state = module.init_simulation(
        n_target, t_init, tr_init, mesh, eos, inv_eos, geometry=geometry
    )

    phase_totals = {'sampling': 0.0, 'transport': 0.0, 'postprocess': 0.0, 'total': 0.0}
    event_totals = {}
    step_count = 0

    wall_start = time.perf_counter()
    while state.time < t_final - 1e-12:
        step_dt = min(dt, t_final - state.time)
        state, info = module.step(
            state, n_target, n_boundary, n_source, n_max,
            t_boundary, step_dt, mesh, sigma_a_f, inv_eos, cv, source,
            reflect=reflect, geometry=geometry, **extra_step_kwargs
        )
        profiling = info.get('profiling', {})
        for key, value in profiling.get('phase_times_s', {}).items():
            phase_totals[key] = phase_totals.get(key, 0.0) + float(value)
        for key, value in profiling.get('transport_events', {}).items():
            if key == 'avg_events_per_particle':
                continue
            event_totals[key] = event_totals.get(key, 0) + int(value)
        step_count += 1
    wall_total = time.perf_counter() - wall_start

    n_transported = max(event_totals.get('n_particles_transported', 0), 1)
    return {
        'method': method_name,
        'steps': step_count,
        'wall_s': wall_total,
        'phase_totals_s': phase_totals,
        'event_totals': event_totals,
        'avg_events_per_particle_overall': event_totals.get('total', 0) / n_transported,
        'final_time': float(state.time),
        'final_particles': int(len(state.weights)),
        'final_total_energy': float(state.previous_total_energy),
    }


def main():
    parser = argparse.ArgumentParser(description='Profile Marshak-wave-like problems with FC and CF solvers.')
    parser.add_argument('--geometry', choices=('slab', 'spherical'), default='slab')
    parser.add_argument('--t-final', type=float, default=1.0)
    parser.add_argument('--dt', type=float, default=0.025)
    parser.add_argument('--cells', type=int, default=50)
    parser.add_argument('--length', type=float, default=0.2)
    parser.add_argument('--n-target', type=int, default=10000)
    parser.add_argument('--n-boundary', type=int, default=10000)
    parser.add_argument('--n-max', type=int, default=40000)
    parser.add_argument('--event-cap-per-particle', type=int, default=0)
    parser.add_argument('--fastpath-threshold', type=float, default=0.0)
    parser.add_argument('--json-out', default='')
    args = parser.parse_args()

    mesh = build_mesh(args.length, args.cells)
    t_init = np.zeros(args.cells) + 1e-4
    tr_init = np.zeros(args.cells) + 1e-4
    source = np.zeros(args.cells)

    if args.geometry == 'slab':
        t_boundary = (1.0, 0.0)
        reflect = (False, True)
    else:
        # Symmetry at the origin, fixed hot bath at the outer radius.
        t_boundary = (0.0, 1.0)
        reflect = (True, False)

    sigma_a_f = lambda t: 300 * t**-3
    cv_val = 0.3
    eos = lambda t: cv_val * t
    inv_eos = lambda u: u / cv_val
    cv = lambda t: cv_val

    fc_result = summarize(
        'FC', fc, mesh, t_init, tr_init, eos, inv_eos, sigma_a_f, cv, source,
        t_boundary, reflect, args.geometry, args.n_target, args.n_boundary,
        0, args.n_max, args.dt, args.t_final
    )
    cf_result = summarize(
        'CF', cf, mesh, t_init, tr_init, eos, inv_eos, sigma_a_f, cv, source,
        t_boundary, reflect, args.geometry, args.n_target, args.n_boundary,
        0, args.n_max, args.dt, args.t_final,
        extra_step_kwargs={
            'event_cap_per_particle': args.event_cap_per_particle,
            'fastpath_threshold': args.fastpath_threshold,
        },
    )

    comparison = {
        'case': {
            'geometry': args.geometry,
            't_final_ns': args.t_final,
            'dt_ns': args.dt,
            'cells': args.cells,
            'domain_cm': args.length,
            'n_target': args.n_target,
            'n_boundary': args.n_boundary,
            'n_max': args.n_max,
            't_boundary': list(t_boundary),
            'reflect': list(reflect),
            'event_cap_per_particle': args.event_cap_per_particle,
            'fastpath_threshold': args.fastpath_threshold,
        },
        'fc': fc_result,
        'cf': cf_result,
        'ratios': {
            'wall': cf_result['wall_s'] / max(fc_result['wall_s'], 1e-30),
            'transport_phase': cf_result['phase_totals_s']['transport'] / max(fc_result['phase_totals_s']['transport'], 1e-30),
            'total_events': cf_result['event_totals'].get('total', 0) / max(fc_result['event_totals'].get('total', 0), 1),
            'avg_events_per_particle': cf_result['avg_events_per_particle_overall'] / max(fc_result['avg_events_per_particle_overall'], 1e-30),
        },
    }

    print(json.dumps(comparison, indent=2, sort_keys=True))
    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as fh:
            json.dump(comparison, fh, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
