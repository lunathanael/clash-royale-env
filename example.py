from interface import Interface
from random import randint # for random action delay

def random_action():
    x = randint(0, 18)
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

for i in range(100):
    interface.start_classic_deck_battle() # Initiate game
    print("Game Requested.")

    while not interface.in_game(): # Wait for game to be in play-ready state
        continue
    print("Game Started!")

    imgs = [] # array to store images
    sum = 0 # random sum to randomize action timings
    while not interface.is_game_over(): # render frames until game over
        imgs.append(interface.get_image())
        sum += randint(1, 2)
        if sum >= 200 - len(imgs) / 10:
            sum = 0
            x, y, card = random_action()
            interface.play_card(x, y, card)

    print("Game outcome:", interface.determine_victor())
    interface.exit_game() # Exit the game-over screen
    print(f"Game had a total of {len(imgs)} frames rendered.")

    rgb_arrays_to_mp4(imgs, f"./replays/game_{i}.mp4")
    print(f"Game video stored at replays/game_{i}.mp4")
