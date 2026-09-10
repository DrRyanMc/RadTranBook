"""Interpolation and macroscopic scaling for Compton multigroup tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .compton import ComptonMatrices
except ImportError:
    from compton import ComptonMatrices


A_RAD_GJ_CM3_KEV4 = 1.3720e-2
C_LIGHT_CM_PER_NS = 29.9792458


class MicroscopicComptonTable:
    """Temperature-interpolated microscopic Compton group data.

    Temperatures are interpolated linearly in log temperature.  Linear matrix
    entries have cm^2 units; induced entries have cm^4 ns GJ^-1 units when
    group scalar intensities use GJ cm^-2 ns^-1.
    """

    def __init__(self, path: str | Path):
        with np.load(path) as data:
            self.energy_edges_kev = np.asarray(data["energy_edges_kev"], dtype=float)
            self.temperature_kev = np.asarray(data["temperature_kev"], dtype=float)
            self.in_scatter_cm2 = np.asarray(data["in_scatter_cm2"], dtype=float)
            self.out_scatter_cm2 = np.asarray(data["out_scatter_cm2"], dtype=float)
            self.in_induced = np.asarray(data["in_induced_cm4_ns_per_gj"], dtype=float)
            self.out_induced = np.asarray(data["out_induced_cm4_ns_per_gj"], dtype=float)
            self.group_mean_energy_kev = (
                np.asarray(data["group_mean_energy_kev"], dtype=float)
                if "group_mean_energy_kev" in data.files else None
            )
            self.group_phase_capacity_kev3 = (
                np.asarray(data["group_phase_capacity_kev3"], dtype=float)
                if "group_phase_capacity_kev3" in data.files else None
            )
            self.group_scalar_capacity = (
                np.asarray(data["group_scalar_capacity_gj_cm2_ns"], dtype=float)
                if "group_scalar_capacity_gj_cm2_ns" in data.files else None
            )
            self.conservative_group_kernel = bool(
                np.asarray(data["conservative_group_kernel"]).item()
            ) if "conservative_group_kernel" in data.files else False
        if self.temperature_kev.ndim != 1 or self.temperature_kev.size < 2:
            raise ValueError("the table must contain at least two temperature nodes")
        groups = self.energy_edges_kev.size - 1
        expected = (self.temperature_kev.size, groups, groups)
        for name in ("in_scatter_cm2", "out_scatter_cm2", "in_induced", "out_induced"):
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}; expected {expected}")
        if self.group_mean_energy_kev is not None:
            if self.group_mean_energy_kev.shape != (groups,):
                raise ValueError(
                    "group_mean_energy_kev must have shape "
                    f"({groups},); got {self.group_mean_energy_kev.shape}"
                )
            if np.any(self.group_mean_energy_kev <= 0.0):
                raise ValueError("group_mean_energy_kev must be positive")
        if self.conservative_group_kernel:
            for name in ("group_phase_capacity_kev3", "group_scalar_capacity"):
                values = getattr(self, name)
                if values is None or values.shape != (groups,) or np.any(values <= 0.0):
                    raise ValueError(
                        f"{name} must be a positive array with shape ({groups},) "
                        "for a conservative group-kernel table"
                    )
        self._log_temperature = np.log(self.temperature_kev)

    @property
    def groups(self) -> int:
        return self.energy_edges_kev.size - 1

    def _bracket(self, temperature_kev: float) -> tuple[int, float]:
        log_temperature = np.log(np.clip(
            temperature_kev, self.temperature_kev[0], self.temperature_kev[-1]
        ))
        upper = int(np.searchsorted(self._log_temperature, log_temperature, side="right"))
        upper = min(max(upper, 1), self.temperature_kev.size - 1)
        lower = upper - 1
        fraction = ((log_temperature - self._log_temperature[lower])
                    / (self._log_temperature[upper] - self._log_temperature[lower]))
        return lower, float(np.clip(fraction, 0.0, 1.0))

    def matrices(
        self, temperature_kev: float, electron_density_cm3: float,
        include_planck_balance: bool = True,
    ) -> ComptonMatrices:
        """Return macroscopic matrices at one temperature.

        The density multiplication is intentionally performed here, leaving
        the stored table independent of a particular material density.
        """
        if electron_density_cm3 < 0.0:
            raise ValueError("electron density must be non-negative")
        bounded_temperature = float(np.clip(
            temperature_kev, self.temperature_kev[0], self.temperature_kev[-1]
        ))
        lower, fraction = self._bracket(bounded_temperature)

        def interpolate(values: np.ndarray) -> np.ndarray:
            return ((1.0 - fraction) * values[lower] + fraction * values[lower + 1])

        in_scatter = electron_density_cm3 * interpolate(self.in_scatter_cm2)
        out_scatter = electron_density_cm3 * interpolate(self.out_scatter_cm2)
        in_induced = electron_density_cm3 * interpolate(self.in_induced)
        out_induced = electron_density_cm3 * interpolate(self.out_induced)
        if self.conservative_group_kernel:
            # Interpolating independently tabulated reverse rates would break
            # their exponential detailed-balance relation between temperature
            # nodes.  Interpolate only the independently supplied downhill
            # kernel and reconstruct all four arrays at the requested T.
            for lower_group in range(self.groups):
                for upper_group in range(lower_group + 1, self.groups):
                    downhill = out_scatter[upper_group, lower_group]
                    reverse_ratio = (
                        self.group_phase_capacity_kev3[upper_group]
                        / self.group_phase_capacity_kev3[lower_group]
                        * np.exp(-(self.group_mean_energy_kev[upper_group]
                                  - self.group_mean_energy_kev[lower_group])
                                 / bounded_temperature)
                    )
                    out_scatter[lower_group, upper_group] = reverse_ratio * downhill
            in_scatter = out_scatter * (
                self.group_mean_energy_kev[None, :]
                / self.group_mean_energy_kev[:, None]
            )
            out_induced = out_scatter / self.group_scalar_capacity[None, :]
            in_induced = in_scatter / self.group_scalar_capacity[None, :]
        planck_net_out = None
        if include_planck_balance:
            planck = self.planck_group_scalar(bounded_temperature)
            # This net opacity is a bounded, group-level detailed-balance
            # normalization for the Planck induced option.  It is used only in
            # that option; Wien, lagged, and nonlinear modes retain the raw
            # tabulated rates.
            planck_net_out = (in_scatter.T @ planck) / np.maximum(planck, 1.0e-300)
        return ComptonMatrices(
            in_scatter=in_scatter,
            out_scatter=out_scatter,
            in_induced=in_induced,
            out_induced=out_induced,
            planck_net_out=planck_net_out,
        )

    def matrices_field(
        self, temperature_kev: np.ndarray, electron_density_cm3: np.ndarray,
        include_planck_balance: bool = False,
    ) -> ComptonMatrices:
        """Return macroscopic data at every point of a temperature field.

        This is the field-valued analogue of matrices.  It is useful for
        spatially varying Compton problems, where each point may lie between a
        different pair of table-temperature nodes.  Conservative kernels
        reconstruct their upward entries after interpolation, which retains
        the temperature-dependent detailed-balance relation.
        """
        temperature = np.asarray(temperature_kev, dtype=float)
        density = np.asarray(electron_density_cm3, dtype=float)
        if temperature.shape != density.shape:
            raise ValueError("temperature and electron-density fields must have the same shape")
        if np.any(density < 0.0):
            raise ValueError("electron density must be non-negative")

        bounded_temperature = np.clip(
            temperature, self.temperature_kev[0], self.temperature_kev[-1]
        )
        log_temperature = np.log(bounded_temperature)
        upper = np.searchsorted(self._log_temperature, log_temperature, side="right")
        upper = np.clip(upper, 1, self.temperature_kev.size - 1)
        lower = upper - 1
        fraction = ((log_temperature - self._log_temperature[lower])
                    / (self._log_temperature[upper] - self._log_temperature[lower]))
        fraction = np.clip(fraction, 0.0, 1.0)

        def interpolate(values: np.ndarray) -> np.ndarray:
            local = ((1.0 - fraction)[..., None, None] * values[lower]
                     + fraction[..., None, None] * values[upper])
            return np.moveaxis(local, (-2, -1), (0, 1)) * density[None, None, ...]

        out_scatter = interpolate(self.out_scatter_cm2)
        if self.conservative_group_kernel:
            for lower_group in range(self.groups):
                for upper_group in range(lower_group + 1, self.groups):
                    delta_energy = (
                        self.group_mean_energy_kev[upper_group]
                        - self.group_mean_energy_kev[lower_group]
                    )
                    reverse_ratio = (
                        self.group_phase_capacity_kev3[upper_group]
                        / self.group_phase_capacity_kev3[lower_group]
                        * np.exp(-delta_energy / bounded_temperature)
                    )
                    out_scatter[lower_group, upper_group] = (
                        reverse_ratio * out_scatter[upper_group, lower_group]
                    )
            in_scatter = None
            in_induced = None
            out_induced = None
        else:
            in_scatter = interpolate(self.in_scatter_cm2)
            in_induced = interpolate(self.in_induced)
            out_induced = interpolate(self.out_induced)

        if include_planck_balance:
            raise NotImplementedError(
                "field-valued Planck normalization is not implemented; "
                "use nonlinear or Wien Compton coupling"
            )
        return ComptonMatrices(
            in_scatter=in_scatter,
            out_scatter=out_scatter,
            in_induced=in_induced,
            out_induced=out_induced,
            planck_net_out=None,
            group_mean_energy=(self.group_mean_energy_kev
                               if self.conservative_group_kernel else None),
            group_scalar_capacity=(self.group_scalar_capacity
                                   if self.conservative_group_kernel else None),
        )

    def planck_group_scalar(self, temperature_kev: float) -> np.ndarray:
        """Return the group scalar Planck intensity, ``4 pi B_g``."""
        nodes, weights = np.polynomial.legendre.leggauss(128)
        result = np.empty(self.groups)
        for group, (lower, upper) in enumerate(zip(
                self.energy_edges_kev[:-1], self.energy_edges_kev[1:])):
            energy = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
            integral = 0.5 * (upper - lower) * np.sum(
                weights * energy**3 / np.expm1(np.minimum(energy / temperature_kev, 700.0))
            )
            result[group] = (
                A_RAD_GJ_CM3_KEV4 * C_LIGHT_CM_PER_NS * 15.0 / np.pi**4
                * integral
            )
        return result
