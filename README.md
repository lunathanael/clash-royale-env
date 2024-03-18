Using Python 3.12.1
```python
python -m pip install -r requirements.txt
```
Reference example.py for showcase of basic functionality.
Reference out.mp4 for example gameplay.
Code to demonstrate an example of:
  1. requesting a game
  2. waiting for game start
  3. storing game frames
  4. applying random actions
  5. determining game state
  6. realizing game end value (including draws)
  7. returning to home screen

```python
from interface import Interface
import numpy as np # for random action 

def random_action():
    # uniform random action
    return np.random.randint(0, 2305)

import imageio
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
    interface.start_classic_deck_battle_friend() # Initiate game
    print("Game Requested.")

    while not interface.in_game(): # Wait for game to be in play-ready state
        continue
    print("Game Started!")

    imgs = [] # array to store images
    sum = 0 # random sum to randomize action timings
    while not interface.is_game_over(): # render frames until game over
        imgs.append(interface.get_image())
        action = random_action()
        if action != 2304:
            x = action // (32*4)
            y = (action % (32*4)) // 4
            card = action % 4
            interface.play_card(x, y, card)

    print("Game outcome:", interface.determine_victor())
    interface.exit_game() # Exit the game-over screen
    print(f"Game had a total of {len(imgs)} frames rendered.")

    rgb_arrays_to_mp4(imgs, f"./replays/game_{i}.mp4")
    print(f"Game video stored at replays/game_{i}.mp4")
```

https://github.com/lunathanael/clash-royale-env/assets/68858103/c12eb88f-afdd-445f-b3ce-de7dd36d255b

