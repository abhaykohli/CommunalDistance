import cv2
import threading
import queue
import time
from config import FPS


class ThreadingClass:
    def __init__(self, source):
        # Initialize input source with cv2
        self.capture = cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # Read FPS from config.py file of project, and assign to self.FPS variable
        self.FPS = 1 / FPS
        self.FPS_MS = int(self.FPS * 1000)

        # This line creates a background thread with update() function as target
        self.thread = threading.Thread(target=self.update, args=())
        # This line makes sure that thread gets closed when program gets closed
        self.thread.daemon = True
        # This line starts background thread
        self.thread.start()

    # This is the core function of this class
    def update(self):
        while True:
            # If input source is available
            if self.capture.isOpened():
                # Read a frame from input source and store in class variable
                (self.status, self.frame) = self.capture.read()
            # Control FPS based on self.FPS
            time.sleep(self.FPS)

    # This gets called from main.py
    def read(self):
        # This simply returns the frame stored previously
        return self.frame
