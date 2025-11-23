# Custom imports
from dem.src.core.factory   import *
from dem.src.material.steel import *
from dem.src.material.glass import *
from dem.src.material.sand  import *

# Declare factory
material_factory = factory()

# Register materials
material_factory.register("steel", steel)
material_factory.register("glass", glass)
material_factory.register("sand",  sand)
