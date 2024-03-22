from android_interface import AndroidInterface
from utilities import vredmean, wait_until, wait_until_nvalue
import numpy as np
import time

class ClanClassicEnv():
    def __init__(self, host: bool):
        self._num_actions = 18 * 32 * 4 + 1
        self._is_host = host
        self._interface = AndroidInterface()
        self._interface.start(True)

    
    def __del__(self):
        self._interface.stop()

    def apply(self, action):
        if self.in_game() and  0 <= action < 2304:
            x = (action // (4 * 32))
            y = (action // 4) % 32
            card = action % 4
            self.play_card(x, y, card)

    def reset(self):
        wait_until(self.on_clan_tab, timeout=120, period=0.1)
        if self._is_host:
            self.start_classic_clan()
        else:
            wait_until(self.pending_clan_battle, timeout=1800, period=0.1)
            self.accept_battle_clan()

        wait_until(self.in_game, timeout=60, period=0.1)

    def get_observation(self):
        return self._interface.get_frame()

    def num_actions(self):
        return self._num_actions
    
    def await_result(self) -> float:
        """
        Waits for a game over result.
        Once a result is determined, the game is exited and result returned.
        """
        wait_until_nvalue(self.result(), nvalue=-1, timeout=100, period=0.1)
        result = self.result()
        self.exit_game()
        return result
    
    def play_card(self, x: int, y: int, card_index: int):
        self._interface.tap(335 + card_index * 200, 2050)

    # action wrappers
    def start_classic_clan(self) -> None:
        self._interface.tap(452, 2054)
        time.sleep(0.05)
        self._interface.swipe(800, 1888, 800, 700, 20)
        time.sleep(0.05)
        self._interface.tap(800, 1250)

    def accept_battle_clan(self) -> None:
        self._interface.tap(855, 1840)

    def exit_game(self) -> None:
        self._interface.tap()

    # detection helpers
    def get_pixel(self, x, y):
        return self._interface.get_frame()[y, x]
    
    def check_pixels(self, coordinates, target_colors, tolerance=20) -> bool:
        coordinates = np.array(coordinates)
        x, y, target_colors = coordinates[:, 0], coordinates[:, 1], np.array(target_colors)
        colors = self.get_pixel(x, y)
        return np.all(vredmean(colors, target_colors) < tolerance)
    
    def pending_clan_battle(self) -> bool:
        """
        Checks for a pending clan battle via yellow button.
        """

        return env.check_pixels([[960, 1860], [750, 1860]], [[49, 189, 252], [49, 189, 252]])
    
    def on_clan_tab(self) -> bool:
        """
        Checks for on clan tab via red x and clan background
        """

        return env.check_pixels([[970, 222], [15, 1080]], [[55, 45, 231], [125, 78, 59]])
    
    def in_game(self) -> bool:
        """
        Checks for in game via elixir, white chat, and card border
        """

        return env.check_pixels([[80, 2000], [210, 2250], [1054, 2256]], [[255, 253, 255], [155, 79, 6], [155, 82, 9]])
    
    def terminal(self) -> bool:
        """
        Checks for terminal state by looking for chat box without elixir/card border
        """

        return env.check_pixels([[80, 2000]], [[255, 254, 255]]) != self.check_pixels([[210, 2250], [1054, 2256]], [[155, 79, 6], [155, 82, 9]])
    
    def result(self) -> float:
        """
        Determines result through checking for text pixel colors
        """
        if env.check_pixels([[421, 1045], [643, 1047], [560, 1040]], [[253, 253, 103], [255, 255, 109], [254, 252, 95]]):
            return 1
        
        if env.check_pixels([[418, 475], [622, 482], [520, 482]], [[251, 200, 254], [255, 204, 252], [255, 211, 255]]):
            return 0
        
        if env.check_pixels([[430, 390], [540, 400], [643, 400]], [[255, 251, 255], [255, 255, 255], [255, 255, 254]]):
            return 0.5
        
        return -1

if __name__ == "__main__":
    # Debugging
    env = ClanClassicEnv(True)
    ls = [[430, 390], [540, 400], [643, 400]]
    for coord in ls:
        print(env.get_pixel(coord[0], coord[1]))

    import scrcpy
    card_index=3
    env._interface.tap(335 + card_index * 200, 2050)
    
    for i in range(18):
        for j in range(32):
            x = 57.6 * i + 50
            y = 46.3 * (31-j) + 280
            env._interface.client.control.touch(x, y, scrcpy.ACTION_DOWN)
            time.sleep(0.7)
            
            env._interface.client.control.touch(335 + card_index * 200, 2050, scrcpy.ACTION_UP)
            time.sleep(0.1)


    print("Checks and states")
    print("Pending clan battle:", env.pending_clan_battle())
    print("On Clan Tab:", env.on_clan_tab())
    print("In game:", env.in_game())
    print("Terminal:", env.terminal())
    print("Result:", env.result())
    del env
