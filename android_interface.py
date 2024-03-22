import scrcpy
import time



class AndroidInterface():

    def __init__(self, serial: str="RFCWC04A2VY"):    
        self.serial = serial
        self.client = scrcpy.Client(device=serial, stay_awake=True,)
        #self.client.add_listener(scrcpy.EVENT_FRAME, self.on_frame)
        #self.frame = None

    def get_frame(self):
        return self.client.last_frame

    def start(self, threaded : bool=True):
        self.client.start(threaded=threaded)

    def stop(self):
        self.client.stop()

    def tap(x, y):
        self.client.control.touch(x, y, scrcpy.ACTION_DOWN)
        self.client.control.touch(x, y, scrcpy.ACTION_UP)


if __name__ == "__main__":
    # Debug Code
    interface = AndroidInterface()
    interface.start(True)
    i = 0
    while interface.get_frame() is None and i < 10000:
        i += 1
        print(i)

    from PIL import Image
    im = Image.fromarray(interface.get_frame())
    im.save("your_file.jpeg")
    print(time.strftime("%H:%M:%S", time.localtime()))
    interface.stop()