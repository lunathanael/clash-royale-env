import subprocess
from typing import Tuple, Dict

import numpy as np
from numpy.typing import NDArray
import win32con
import win32gui
import win32ui


def bmp_to_rgb(bmpinfo: dict, bmpstr: bytes) -> NDArray[np.uint8]:
    """
    Converts a bitmap into an RGB image.

    Args:
        bmpinfo (dict): Information about the bitmap, including width and height.
        bmpstr (bytes): The bitmap data as a bytes object.

    Returns:
        NDArray[np.uint8]: The resulting image as an RGB numpy array.
    """
    img = np.frombuffer(bmpstr, dtype='uint8')
    img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
    img_bgr = img[:, :, :3]
    return img_bgr[:, :, ::-1]


class Interface:
    """
    Interface class to interact with the game application, allowing for operations such as
    taking screenshots, simulating game actions, and analyzing game states.

    This class is designed to facilitate automation tasks within a specific application window
    by leveraging the Windows API for graphical operations and subprocesses for executing utilities.

    Attributes:
        w (int): Width of the capture area.
        h (int): Height of the capture area.
        xpos (int): X-coordinate of the top-left corner of the capture area.
        ypos (int): Y-coordinate of the top-left corner of the capture area.
    """

    def __init__(self, w: int = 414, h: int = 736, x: int = 0, y: int = 0) -> None:
        """
        Initializes the interface with specified dimensions and position.

        Args:
            w (int): Width of the window.
            h (int): Height of the window.
            x (int): X-coordinate of the window's top-left corner.
            y (int): Y-coordinate of the window's top-left corner.
        """

        subprocess.run("./utilities/init_window.exe")
        self.w: int = w
        self.h: int = h
        self.xpos: int = x + 11  # Adjusted for window decorations
        self.ypos: int = y + 45  # Adjusted for window decorations

    def screenshot_window(self) -> Tuple[Dict[str, int], bytes]:
        """
        Captures a screenshot of the specified area of the desktop.

        Returns:
            Tuple[dict, bytes]: Bitmap information and the bitmap data.
        """

        hdesktop: int = win32gui.GetDesktopWindow()
        hwndDC: int = win32gui.GetWindowDC(hdesktop)
        mfcDC: win32ui.CDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC: win32ui.CDC = mfcDC.CreateCompatibleDC()

        saveBitMap: win32ui.CBitmap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, self.w, self.h)

        saveDC.SelectObject(saveBitMap)

        saveDC.BitBlt((0, 0), (self.w, self.h), mfcDC, (self.xpos, self.ypos), win32con.SRCCOPY)

        bmpinfo: Dict[str, int] = saveBitMap.GetInfo()
        bmpstr: bytes = saveBitMap.GetBitmapBits(True)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hdesktop, hwndDC)

        return bmpinfo, bmpstr

    def get_image(self) -> NDArray[np.uint8]:
        """
        Obtains the current screenshot as an RGB image.

        Returns:
            NDArray[np.uint8]: The image captured from the specified area. Shape: (height, width, channels).
        """

        bmpinfo, bmpstr = self.screenshot_window()
        return bmp_to_rgb(bmpinfo, bmpstr)

    def get_pixel_color(self, x: int, y: int) -> Tuple[int, int, int]:
        """
        Get the color of the pixel at the given (x, y) coordinates.

        Args:
            x (int): The relative x-coordinate of the pixel.
            y (int): The relative y-coordinate of the pixel.

        Returns:
            Tuple[int, int, int]: A tuple (r, g, b) representing the color of the pixel.
        """
        
        hdc = win32gui.GetDC(None)
        pixel_color: int = win32gui.GetPixel(hdc, self.xpos + x, self.ypos + y)
        win32gui.ReleaseDC(None, hdc)
        r: int = pixel_color & 0xff
        g: int = (pixel_color >> 8) & 0xff
        b: int = (pixel_color >> 16) & 0xff
        return (r, g, b)

    def play_card(self, x: int, y: int, card_index: int) -> None:
        """
        Simulates playing a card action within the game interface.

        Args:
            x (int): The x-coordinate where to play the card.
            y (int): The y-coordinate where to play the card.
            card_index (int): The index of the card to be played.
        """
        dx, dy = x, 31 - y
        xcoord = 38 + dx * 20
        ycoord = 65 + dy * 16
        cardx, cardy = 130 + 77 * card_index, 650
        subprocess.run(["./utilities/play_card.exe", str(xcoord), str(ycoord), str(cardx), str(cardy)])

    def is_game_over(self) -> bool:
        """
        Determines whether the game has concluded, specifically, if the 'ok" button prompt is available.

        Returns:
            bool: True if the game is over, otherwise False.
        """
        #return (self.get_pixel_color(172, 643) == (82, 160, 224)) and (self.get_pixel_color(234, 641) == (91, 164, 224)) and (self.get_pixel_color(203, 643) == (224, 224, 224))
        return self.determine_victor() != -1

    def determine_victor(self) -> float:
        """
        Determines the outcome of the game.

        Returns:
            float: A numerical representation of the game outcome (0 for opponent win, 0.5 for draw, 1 for player win).
            Error states will return -1.
        """

        if (self.get_pixel_color(163, 49) == (224, 224, 224)) and (self.get_pixel_color(193, 51) == (224, 224, 224)) and (self.get_pixel_color(248, 58) == (224, 224, 224)):
            return 0.5
        if (self.get_pixel_color(156, 75) == (224, 179, 224)) and (self.get_pixel_color(213, 86) == (224, 179, 224)) and (self.get_pixel_color(237, 82) == (224, 179, 224)):
            return 0
        if (self.get_pixel_color(161, 302) == (90, 224, 224)) and (self.get_pixel_color(207, 294) == (90, 224, 224)) and (self.get_pixel_color(246, 301) == (90, 224, 224)):
            return 1
        
        #raise RuntimeWarning("No Gameover event detected! Make sure this is intended behavior, otherwise, report this as a bug. Returning -1.")
        return -1

    def in_game(self) -> bool:
        """
        Checks if the player is currently in a game by looking for the bottom blue card area.
        Careful, a False value doesn't guarantee the user is at a specific state, only that no actions can be played.

        Returns:
            bool: True if the player is in a game, otherwise False.
        """
        if (self.get_pixel_color(81, 702) == (8, 83, 157)) and (self.get_pixel_color(402, 700) == (8, 86, 161)):
            return True
        return False
    
    def accept_battle(self) -> None:
        """
        Accepts a friendly challenge from the home menu.
        """
        subprocess.run("./utilities/accept_battle.exe")

    def exit_game(self) -> None:
        """
        Exits a finished game by pressing 'ok'.
        """
        subprocess.run("./utilities/exit_game.exe")

    def start_classic_deck_battle(self) -> None:
        """
        Initiates a classic deck battle within the game interface.
        """

        subprocess.run("./utilities/start_classic_deck.exe")
