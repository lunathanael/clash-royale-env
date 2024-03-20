from interface import Interface
import time

interface = Interface() # Start the game interface
print("Interface Initialized.")

ls = [(188, 335)]
for x in ls:
    print(interface.get_pixel_color(x[0], x[1]), end=', ')