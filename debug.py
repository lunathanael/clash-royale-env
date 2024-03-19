from interface import Interface

interface = Interface() # Start the game interface
print("Interface Initialized.")


print(interface.on_clan_tab())
ls = [(30, 49), (151, 632), (373, 43)]
for x in ls:
    print(interface.get_pixel_color(x[0], x[1]), end=', ')

from network import uniform_policy
print(uniform_policy(1))