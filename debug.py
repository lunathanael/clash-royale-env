from interface import Interface
import time

interface = Interface() # Start the game interface
print("Interface Initialized.")

print(interface.in_game())
ls = [(62, 425), (16, 390)]
for x in ls:
    print(interface.get_pixel_color(x[0], x[1]), end=', ')