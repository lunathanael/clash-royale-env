import scrcpy
from utilities import wait_until


class AndroidInterface():

    def __init__(self, serial: str="RFCWC04A2VY"):
        self.serial = serial
        self.client = scrcpy.Client(device=serial, stay_awake=True,)

    def get_frame(self):
        return self.client.last_frame

    def ready(self) -> bool:
        return self.get_frame() is not None

    def start(self, threaded : bool=True) -> None:
        self.client.start(threaded=threaded)
        wait_until(self.ready, timeout=20, period=0.1)

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

    from PIL import Image
    im = Image.fromarray(interface.get_frame())
    im.save("temp.jpeg")
    import time
    print(time.strftime("%H:%M:%S", time.localtime()))
    interface.tap(452, 2054)
    time.sleep(0.05)
    interface.swipe(800, 1888, 800, 700, 20)
    time.sleep(0.05)
    interface.tap(800, 1200)
    interface.stop()