import sys
import time
import threading


class Spinner:
    def __init__(self, text):
        self.text = text
        self.isRunning = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.animate)
        self.thread.start()
    
    def animate(self):
        pattern = [".", "..", "..."]
        i = 0
        while self.running:
            sys.stdout.write(f"\r\033[K\033[32m{self.text}{pattern[i % len(pattern)]}\033[0m")
            sys.stdout.flush()
            i += 1
            time.sleep(0.3)
    
    def stop(self):
        self.running = False
        self.thread.join()
        sys.stdout.write(f"\r\n\033[92m{self.text} complete!\033[0m\n")
        sys.stdout.flush()