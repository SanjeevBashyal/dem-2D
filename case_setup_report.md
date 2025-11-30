# Case Setup Information for Sediment Transport Simulations

This document summarizes the simulation parameters, mesh resolution, and boundary conditions used in the sediment transport applications (`sediment_transport_k_omega_standard.py`, `sediment_transport_k_omega.py`, `sediment_transport_fvm_ibm.py`).

## 1. Simulation Cases

Two flow regimes were simulated:

| Case Name | Slope | Inlet Velocity ($U_{inlet}$) | Flow Regime |
| :--- | :--- | :--- | :--- |
| **Sub-Critical** | 0.01 | **1.0 m/s** | Sub-critical flow |
| **Super-Critical** | 0.02 | **1.5 m/s** | Super-critical flow |

## 2. Domain and Mesh Resolution

The computational domain and grid resolution are consistent across all cases:

*   **Domain Length ($L$)**: 2.0 m
*   **Domain Width ($W$)**: 0.1 m
*   **Domain Height ($H$)**: 0.4 m
*   **Number of Cells in X ($N_x$)**: 200
*   **Number of Cells in Y ($N_y$)**: 40
*   **Grid Spacing ($dx$)**: 0.01 m
*   **Grid Spacing ($dy$)**: 0.01 m

## 3. Initial Conditions and Particles

*   **Initial Water Depth ($h_{water}$)**: 0.2 m
*   **Particle Size**: 0.02 m (2 cm)
*   **Particle Density**: 2500 kg/m³
*   **Bed Initialization**: Randomly packed bed of approximately 60 particles located between $x=0.5$ m and $x=1.5$ m.

## 4. Boundary Conditions

### Inlet (Left Boundary, $x=0$)
*   **Velocity**: Fixed at $U_{inlet}$ for the water phase ($y \le h_{inlet}$), zero for air.
*   **Phase Fraction ($\alpha$)**: Fixed at 1.0 (water) for $y \le h_{inlet}$, 0.0 (air) above.
*   **Turbulence**:
    *   $k_{inlet} = 1.5 (I \cdot U_{inlet})^2$, where intensity $I = 0.05$.
    *   $\epsilon_{inlet}$ or $\omega_{inlet}$ derived from mixing length $L_{scale} = 0.07 h_{inlet}$.

### Outlet (Right Boundary, $x=L$)
*   **Pressure**: Fixed hydrostatic pressure distribution corresponding to the initial water depth $h_{water}$.
*   **Velocity**: Zero gradient (Neumann).
*   **Phase Fraction**: Zero gradient.

### Walls
*   **Bottom ($y=0$)**: No-slip wall for fluid (with wall functions for turbulence).
*   **Top ($y=H$)**: Free slip / Open boundary.
