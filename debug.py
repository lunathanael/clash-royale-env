from interface import Interface
import time

interface = Interface() # Start the game interface
print("Interface Initialized.")

print(interface.on_clan_tab())
ls = [(4, 293), (2, 103)]
for x in ls:
    print(interface.get_pixel_color(x[0], x[1]), end=', ')