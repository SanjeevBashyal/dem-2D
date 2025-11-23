from numba import float64
from numba.experimental import jitclass

### ************************************************
### Class defining sand material
spec = [
    ('density', float64),
    ('young',   float64),
    ('poisson', float64),
    ('mu_wall', float64),
    ('mu_part', float64),
    ('e_wall',  float64),
    ('e_part',  float64),
    ('Y',       float64),
    ('G',       float64)
]
@jitclass(spec)
class sand():
    ### ************************************************
    ### Constructor
    def __init__(self):

        # Density (kg/m3)
        self.density = 2600.0

        # Young modulus (N/m2) - Quartz
        self.young   = 70.0e9

        # Poisson ratio (unitless)
        self.poisson = 0.25

        # Static friction coeff. on a wall (unitless)
        self.mu_wall = 0.4

        # Static friction coeff. on a particle (unitless)
        self.mu_part = 0.5

        # Restitution coeff. on a wall (unitless)
        self.e_wall = 0.5

        # Restitution coeff. on a particle (unitless)
        self.e_part = 0.6

        # Effective young modulus
        self.Y = (1.0 - self.poisson**2)/self.young

        # Effective shear modulus
        self.G = 2.0*(2.0 + self.poisson)*(1.0 - self.poisson)/self.young
