from interface import Interface

class ClassicEnv():
    def __init__(self, host: bool):
        self._num_actions = 18 * 32 * 4 + 1
        self._is_host = host
        self._interface = Interface()
        self._interface.navigate_clan_tab()

    def apply(self, action):
        if 0 <= action < 2304:
            x = (action // (4 * 32))
            y = (action // 4) % 32
            card = action % 4
            self._interface.play_card(x, y, card)

    def reset(self):
        while not self._interface.on_clan_tab():
            continue
        if self._is_host:
            self._interface.start_classic_deck_battle_clan()
        else:
            while not self._interface.pending_clan_battle():
                continue
            self._interface.accept_battle_clan()

        while not self._interface.in_game():
            continue

    def get_observation(self):
        return self._interface.get_image()

    def terminal(self):
        return self._interface.is_terminal()
    
    
    def result(self):
        result = -1
        while result == -1:
            result = self._interface.determine_victor()
        self._interface.exit_game()
        return result

    def num_actions(self):
        return self._num_actions
