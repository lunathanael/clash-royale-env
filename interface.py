import subprocess
from typing import List, Tuple, Dict

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

    def __init__(self, w: int = 265, h: int = 471, x: int = 0, y: int = 0) -> None:
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
        self.xpos: int = x + 9  # Adjusted for window decorations
        self.ypos: int = y + 38  # Adjusted for window decorations

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

    def play_card(self, x: int, y: int, card_index: int, block: bool=True) -> None:
        """
        Simulates playing a card action within the game interface.

        Args:
            x (int): The x-coordinate where to play the card.
            y (int): The y-coordinate where to play the card.
            card_index (int): The index of the card to be played.
            block (bool): True value will block the function from completing until the action is made.
        """
        dx, dy = x, 31 - y
        xcoord = 27 + dx * 12.6
        ycoord = 43 + dy * 10
        cardx, cardy = 80 + 53 * card_index, 400
        if block:
            subprocess.run(["./utilities/play_card.exe", str(xcoord), str(ycoord), str(cardx), str(cardy)])
        else:
            subprocess.Popen(["./utilities/play_card.exe", str(xcoord), str(ycoord), str(cardx), str(cardy)])

    def check_pixels(self, coordinates: List[Tuple[int]], target: List[Tuple[int]]) -> bool:
        """
        Function to check a list of pixels against the target RGB values, performs absolute comparisons.

        Args:
            coordinates: (List[Tuple[int]]) List of coordinates in the form of (x, y) to examine.
            target: (List[Tuple[int]]) List of target RGB values in the form of (R, G, B) to compare with.

        Returns:
            bool: If all coordinates match the target.
        """
        for idx, coordinate in enumerate(coordinates):
            if self.get_pixel_color(coordinate[0], coordinate[1]) != target[idx]:
                return False
        return True

    def find_pixel_mismatch(self, coordinates: List[Tuple[int]], target: List[Tuple[int]]) -> Tuple[int] | None:
        """
        Function to check a list of pixels against the target RGB values, performs absolute comparisons.

        Args:
            coordinates: (List[Tuple[int]]) List of coordinates in the form of (x, y) to examine.
            target: (List[Tuple[int]]) List of target RGB values in the form of (R, G, B) to compare with.

        Returns:
            Tuple[int]: Returns the coordinate that failed the check, or None if none were found.
        """
        for idx, coordinate in enumerate(coordinates):
            if self.get_pixel_color(coordinate[0], coordinate[1]) != target[idx]:
                return coordinate
        return None

    def is_terminal(self) -> bool:
        """
        Determines whether the game is terminal, specifically, if the card frame is still available.
        Returns:
            bool: True if the game is terminal, otherwise False.
        """
        return not self.in_game()

    def on_clan_tab(self) -> bool:
        """
        Determines if the interface is on the clan tab, specifically by looking for the red exit box.
        Returns:
            bool: True if on clan tab, otherwise False.
        """
        return self.check_pixels([(90, 400), (240, 27)], [(183, 98, 254), (255, 255, 255)])


    def determine_victor(self) -> float:
        """
        Determines the outcome of the game.

        Returns:
            float: A numerical representation of the game outcome (0 for opponent win, 0.5 for draw, 1 for player win).
            Error states will return -1.
        """
        if self.check_pixels([(107, 34), (136, 36), (160, 37)], [(255, 255, 255), (255, 255, 255), (255, 255, 255)]):
            return 0.5
        if self.check_pixels([(102, 55), (137, 55), (158, 55)], [(255, 204, 255), (255, 204, 255), (255, 204, 255)]):
            return 0
        if self.check_pixels([(102, 191), (127, 194), (154, 193)], [(102, 255, 255), (102, 255, 255), (102, 255, 255)]):
            return 1

        #raise RuntimeWarning("No Gameover event detected! Make sure this is intended behavior, otherwise, report this as a bug. Returning -1.")
        return -1

    def in_game(self) -> bool:
        """
        Checks if the player is currently in a game by checking for two pixels in chat box.
        A False value doesn't guarantee the user is at a specific state, only that no actions can be played.
        Note: Seemingly after 8PM EST, the color scheme changes for the game.

        Returns:
            bool: True if the player is in a game, otherwise False.
        """
        return self.check_pixels([(62, 425), (16, 390)], [(37, 41, 45), (255, 255, 255)])

    def pending_clan_battle(self) -> bool:
        """
        Checks if there is currently a pending clan battle challenge, must be on clan tab, only checks 1 pixel.

        Returns:
            bool: True if the player is in a game, otherwise False.
        """
        return self.check_pixels([(183, 336)], [(255, 202, 81)])

    def accept_battle_friend(self) -> None:
        """
        Accepts a friendly challenge from the home menu.
        """
        subprocess.run("./utilities/accept_battle_friend.exe")

    def accept_battle_clan(self) -> None:
        """
        Accepts a clan challenge from clan tab.
        Assumes clan tab is open and challenge is at the bottom
        """
        subprocess.run("./utilities/accept_battle_clan.exe")

    def exit_game(self) -> None:
        """
        Exits a finished game by pressing 'ok'.
        """
        subprocess.run("./utilities/exit_game.exe")

    def start_classic_deck_battle_clan(self) -> None:
        """
        Initiates a classic deck battle within the game interface in the clan.
        Assumes the clan tab is open
        """

        subprocess.run("./utilities/start_classic_deck_clan.exe")


    def navigate_clan_tab(self) -> None:
        """
        Navigates to the clan tab from the home tab.
        """

        subprocess.run("./utilities/navigate_clan_tab.exe")
