"""
LNG Heat Exchanger Freezing Simulation Package
===============================================

Simulates CO2 freeze-out in shell-and-tube LNG heat exchangers.
The primary observable for freezing detection is the tube-side pressure
drop (delta P), which rises nonlinearly as frost narrows the hydraulic
diameter.

Modules
-------
correlations   : Heat transfer and pressure drop correlations (Dittus-Boelter,
                 Churchill 1977, Kern 1950) for shell-and-tube geometry.
freezing_model : CO2 frost deposition / erosion kinetics and wall temperature
                 from the thermal resistance network.
heat_exchanger : 1-D transient PDE solver (Method of Lines + scipy solve_ivp).
scenarios      : Operating scenario definitions and sensor noise injection.
"""

try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("heat_exchanger_monitoring")
except Exception:  # pragma: no cover
    __version__ = "0.1.0-dev"

__all__ = [
    "correlations",
    "freezing_model",
    "heat_exchanger",
    "scenarios",
]
