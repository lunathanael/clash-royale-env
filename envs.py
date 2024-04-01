from android_interface import AndroidInterface
from utilities import vredmean, wait_until, wait_until_nvalue
import numpy as np
import time

class ClanClassicEnv():
    def __init__(self, serial: str, host: bool):
        self._num_actions = 18 * 32 * 4 + 1
        self._is_host = host
        self._interface = AndroidInterface(serial=serial)
        self._interface.start(threaded=False)

    
    def __del__(self):
        self._interface.stop()

    def apply(self, action):
        if self.in_game() and (0 <= action < 2304):
            x = (action // (4 * 32))
            y = (action // 4) % 32
            card = action % 4
            self.play_card(x, y, card)

    def reset(self):
        wait_until(self.on_clan_tab, timeout=120, period=0.05)
        if self._is_host:
            self.start_classic_clan()
        else:
            wait_until(self.pending_clan_battle, timeout=1800, period=0.05)
            self.accept_battle_clan()
        print("Waiting")
        wait_until(self.in_game, timeout=1800, period=0.05)
        print("In game")

    def get_observation(self):
        return self._interface.get_frame()

    def num_actions(self):
        return self._num_actions
    
    def await_result(self) -> float:
        """
        Waits for a game over result.
        Once a result is determined, the game is exited and result returned.
        """
        wait_until_nvalue(self.result, nvalue=-1, timeout=100, period=0.05)
        result = self.result()
        self.exit_game()
        return result
    
    def play_card(self, x: int, y: int, card_index: int):
        self._interface.tap(335 + card_index * 200, 2050)
        x = 57.6 * x + 50
        y = 46.3 * (31-y) + 280
        self._interface.tap(x, y)

    # action wrappers
    def start_classic_clan(self) -> None:
        self._interface.tap(452, 2054)
        time.sleep(0.3)
        self._interface.swipe(800, 1888, 800, 1600, move_step_length=20)
        time.sleep(0.5)
        self._interface.tap(800, 1888)

    def accept_battle_clan(self) -> None:
        self._interface.tap(855, 1840)

    def exit_game(self) -> None:
        self._interface.tap(540, 1940)

    # detection helpers
    def get_pixel(self, x, y):
        return self._interface.get_frame()[y, x]
    
    def check_pixels(self, coordinates, target_colors, tolerance=30) -> bool:
        coordinates = np.array(coordinates)
        x, y, target_colors = coordinates[:, 0], coordinates[:, 1], np.array(target_colors)
        colors = self.get_pixel(x, y)
        return np.all(vredmean(colors, target_colors) < tolerance)
    
    def pending_clan_battle(self) -> bool:
        """
        Checks for a pending clan battle via yellow button.
        """

        return self.check_pixels([[960, 1860], [750, 1860]], [[49, 189, 252], [49, 189, 252]])
    
    def on_clan_tab(self) -> bool:
        """
        Checks for on clan tab via red x and clan background
        """

        return self.check_pixels([[970, 222], [15, 1080]], [[55, 45, 231], [125, 78, 59]])
    
    def in_game(self) -> bool:
        """
        Checks for in game via elixir, white chat, and card border
        """

        return self.check_pixels([[120, 2000], [246, 2309], [1041, 2256]], [[253, 253, 253], [189, 26, 192], [160, 84, 11]])
    
    def terminal(self) -> bool:
        """
        Checks for terminal state by looking for chat box without elixir/card border
        """

        return self.check_pixels([[120, 2000]], [[253, 253, 253]]) != self.check_pixels([[246, 2309], [1041, 2256]], [[189, 26, 192], [160, 84, 11]])
    
    def result(self) -> float:
        """
        Determines result through checking for text pixel colors
        """
        if self.check_pixels([[421, 1045], [643, 1047], [560, 1040]], [[253, 253, 103], [255, 255, 109], [254, 252, 95]]):
            return 1
        
        if self.check_pixels([[418, 475], [622, 482], [520, 482]], [[251, 200, 254], [255, 204, 252], [255, 211, 255]]):
            return 0
        
        if self.check_pixels([[430, 390], [540, 400], [643, 400]], [[255, 251, 255], [255, 255, 255], [255, 255, 254]]):
            return 0.5
        
        return -1

if __name__ == "__main__":
    # Debugging
    env = ClanClassicEnv(True)



    # from cProfile import Profile
    # from pstats import SortKey, Stats

    # print("Profling started:")
    # with Profile() as profile:
    #  print(f"{wait_until(env.on_clan_tab, timeout=10) = }")
    #  (
    #      Stats(profile)
    #      .strip_dirs()
    #      .sort_stats(SortKey.TIME)
    #      .print_stats()
    #  )

    # ls = [[970, 222], [15, 1080]]
    # for coord in ls:
    #     print(env.get_pixel(coord[0], coord[1]))

    #env.start_classic_clan()
    while True:
        assert env.get_observation().shape == (2336, 1080, 3)
    #wait_until(env.pending_clan_battle, timeout=1000)

    print("Checks and states")
    print("Pending clan battle:", env.pending_clan_battle())
    print("On Clan Tab:", env.on_clan_tab())
    print("In game:", env.in_game())
    print("Terminal:", env.terminal())
    print("Result:", env.result())
    del env
