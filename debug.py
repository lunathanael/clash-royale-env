from interface import Interface
import time

interface = Interface() # Start the game interface
print("Interface Initialized.")

time.sleep(2)

for i in range(32):
    interface.play_card(5, i, 0)
    time.sleep(0.5)

ls = [(102, 191), (127, 194), (154, 193)]
for x in ls:
    print(interface.get_pixel_color(x[0], x[1]), end=', ')