import sys
import threading
import time

class LoadingIndicator:
    def __init__(self, message="Processando"):
        self.message = message
        self.is_running = False
        self.animation_thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.is_running = True
        self.animation_thread = threading.Thread(target=self._animate)
        self.animation_thread.daemon = True
        self.animation_thread.start()

    def stop(self):
        self.is_running = False
        if self.animation_thread:
            self.animation_thread.join()
        # Clear the entire line to ensure no loading animation characters remain
        sys.stdout.write('\r' + ' ' * (len(self.message) + 50) + '\r')
        sys.stdout.flush()

    def _animate(self):
        animation = "|/-\\"
        idx = 0
        while self.is_running:
            progress = animation[idx % len(animation)]
            sys.stdout.write(f'\r{self.message} {progress}')
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)
