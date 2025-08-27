import sys
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets

from AquilaWindow import AquilaWindow
from MenuScreen import MenuScreen, Particle



# --------------
# Main App Class
# --------------

class App(QtWidgets.QMainWindow):
    def __init__(self, logo_path="", icon_path=""):
        super().__init__()
        self.setWindowTitle("AQUILA")
        self.resize(1080, 720)

        if icon_path:
            qicon = QtGui.QIcon(icon_path)

        if sys.platform.startswith("win"): # For Windows icon loading
            if not qicon.isNull():
                self.setWindowIcon(qicon)
        elif sys.platform.startswith("darwin") and icon_path: # For macOS icon loading
            # Note: requires PyObjC to be installed in the Python environment
            try:
                import importlib
                AppKit = importlib.import_module("AppKit")
                NSApplication = AppKit.NSApplication
                NSImage = AppKit.NSImage
                img = NSImage.alloc().initWithContentsOfFile_(str(icon_path))
                if img:
                    NSApplication.sharedApplication().setApplicationIconImage_(img)
            except Exception:
                # PyObjC not installed or icon load failed; ignore
                print("Warning: Could not set application icon on macOS. Ensure PyObjC is installed.")
                pass


        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self.menu = MenuScreen(logo_path=logo_path)
        self.params = AquilaWindow()

        self.stack.addWidget(self.menu)    # index 0
        self.stack.addWidget(self.params)  # index 1

        # Wire menu actions
        self.menu.startRequested.connect(self._go_run)
        self.menu.settingsRequested.connect(self._go_settings)
        self.menu.quitRequested.connect(self.close)

        # Optional: menu bar or status bar
        self.statusBar().showMessage("Status: READY - Let's get cooking...")

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)  # Higher = lower FPS = slower particles

        self._particles = []
        self.setMouseTracking(True)
        self._mouse = QtCore.QPointF(-1000, -1000)

    def _go_run(self):
        # Jump straight to the parameters window (or trigger a default run)
        self.stack.setCurrentWidget(self.params)

    def _go_settings(self):
        # If you separate a settings screen, push that here
        self.stack.setCurrentWidget(self.params)

    def _tick(self):
        for p in self._particles:
            p.update()
        self.update()

    def showEvent(self, e):
        super().showEvent(e)
        self._generate_particles()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        self._mouse = QtCore.QPointF(e.position())
        super().mouseMoveEvent(e)

    def _generate_particles(self):
        # Keep a moderate amount for performance
        count = 250
        rect = QtCore.QRectF(0, 0, self.width(), self.height())
        self._particles = [Particle(rect, max_alpha=100) for _ in range(count)]

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        for part in self._particles:
            part.draw(p, self._mouse)


def main():
    app = QtWidgets.QApplication(sys.argv)

    # Parent directory of this script
    base_dir = Path(__file__).resolve().parent

    # Assets are in the 'assets' folder within the parent directory
    assets_dir = base_dir / "assets"
    logo = str(assets_dir / "aquila_full.png")
    icon = str(assets_dir / "aquila_logo.ico")

    win = App(logo_path=logo, icon_path=icon)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
