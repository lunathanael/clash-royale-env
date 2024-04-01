import scrcpy
from utilities import wait_until


class AndroidInterface():

    def __init__(self, serial: str):
        self.serial = serial
        self.client = scrcpy.Client(device=serial, stay_awake=True,)

    def get_frame(self):
        self.client.update_frame()
        return self.client.last_frame

    def start(self, threaded : bool=True) -> None:
        self.client.start(threaded=threaded)

    def stop(self) -> None:
        self.client.stop()

    def tap(self, x: int, y: int) -> None:
        """
        Tap screen

        Args:
            x: horizontal position
            y: vertical position
        """
        self.client.control.touch(x, y, scrcpy.ACTION_DOWN)
        self.client.control.touch(x, y, scrcpy.ACTION_UP)
        
    def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        move_step_length: int = 5,
        move_steps_delay: float = 0.005,
    ) -> None:
        """
        Swipe on screen

        Args:
            start_x: start horizontal position
            start_y: start vertical position
            end_x: start horizontal position
            end_y: end vertical position
            move_step_length: length per step
            move_steps_delay: sleep seconds after each step
        :return:
        """
        self.client.control.swipe(start_x, start_y, end_x, end_y, move_step_length, move_steps_delay)


if __name__ == "__main__":
    # Debug Code
    interface = AndroidInterface()
    interface.start(True)

    lf = interface.get_frame()
    while True:
        if (lf != interface.get_frame()).any():
            print("New Frame")
            lf = interface.get_frame()

    interface.stop()