from interface import Interface
from random import randint # for random action delay

def random_action():
    x = randint(0, 17)
    y = randint(0, 31)
    card = randint(0, 3)
    return x, y, card

import imageio
import numpy as np
def rgb_arrays_to_mp4(rgb_arrays, output_path, fps: int = 30):
    # Ensure the output_path ends with .mp4
    if not output_path.endswith('.mp4'):
        output_path += '.mp4'
    
    # Create a writer object specifying the output file path and fps
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=9)
    
    # Iterate over each frame in the array of RGB arrays
    for frame in rgb_arrays:
        # Write the frame to the video
        writer.append_data(frame)
    
    # Close the writer to finalize the video file       
    writer.close()

interface = Interface() # Start the game interface
print("Interface Initialized.")
print(interface.on_clan_tab())
ls = [(30, 49), (151, 632), (373, 43)]
for x in ls:
    print(interface.get_pixel_color(x[0], x[1]), end=', ')

from network import uniform_policy
print(uniform_policy(1))