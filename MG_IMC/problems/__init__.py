"""
MG_IMC problem definitions.

Each submodule defines a benchmark problem with its geometry, material
properties, opacity model, initial and boundary conditions.

Available problems
------------------
dilute_spectrum_shell
    1-D spherical shell problem with dilute, non-Planckian radiation spectrum.
marshak_wave_powerlaw
    1-D slab Marshak wave with power-law opacity.  Includes analytic reference.
su_olson_picket_fence
    Two-group Su-Olson picket-fence benchmark with analytic reference data.
crooked_pipe_multigroup_imc
    2-D multigroup IMC crooked-pipe radiation transport problem.
mg_imc1d_spherical
    Gray-equivalent spherical test for MG_IMC1D.
infinite_medium_comparison
    0-D infinite-medium comparison between MG-IMC and scipy reference.
"""
