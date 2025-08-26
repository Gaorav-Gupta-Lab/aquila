import sys, random, math
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets

from aquila_utils import AquilaWindow


# ---------------------------------------- 
# Particle System for Background Animation 
# ----------------------------------------

class Particle:
    def __init__(self, rect: QtCore.QRectF, max_alpha=245):
        self.bounds = rect
        self.pos = QtCore.QPointF(
            random.uniform(rect.left()+5, rect.right()-5),
            random.uniform(rect.top()+5, rect.bottom()-5)
        )
        self.vel = QtCore.QPointF(
            random.uniform(-0.8, 0.8),
            random.uniform(-0.8, 0.8)
        )
        self.radius = random.uniform(2.0, 3.5)
        self.maxd = 150.0
        self.max_alpha = max_alpha
        self.alpha = random.randint(15, max_alpha)
        self.alpha_multiplier = random.choice([-1, 1])

    def update(self):
        p = self.pos
        v = self.vel
        b = self.bounds

        p.setX(p.x() + v.x())
        p.setY(p.y() + v.y())

        # bounce on edges
        if p.x() < b.left()+5 or p.x() > b.right()-5:
            self.vel.setX(-self.vel.x())
        if p.y() < b.top()+5 or p.y() > b.bottom()-5:
            self.vel.setY(-self.vel.y())

        # "flickering" effect
        self.alpha += 1 * self.alpha_multiplier
        if self.alpha < 10:
            self.alpha_multiplier *= -1
        if self.alpha > self.max_alpha:
            self.alpha_multiplier *= -1

    def draw(self, painter: QtGui.QPainter, mouse: QtCore.QPointF):
        # dot
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(255, 255, 255, self.alpha))
        painter.drawEllipse(self.pos, self.radius, self.radius)

        # faint line towards mouse if near
        dx = self.pos.x() - mouse.x()
        dy = self.pos.y() - mouse.y()
        d2 = dx*dx + dy*dy
        if d2 < self.maxd*self.maxd:
            d = max(1.0, math.sqrt(d2))
            t = 1.0 - (d / self.maxd)
            t = t*t  # ease
            a = int(160 * t)
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255, a), 2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawLine(self.pos, mouse)


# ----------------------------------------------- 
# Menu Screen (painted background, logo, buttons) 
# -----------------------------------------------

class MenuScreen(QtWidgets.QWidget):
    startRequested = QtCore.Signal()
    settingsRequested = QtCore.Signal()
    quitRequested = QtCore.Signal()

    def __init__(self, logo_path: str = "", parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._mouse = QtCore.QPointF(-1000, -1000)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(20)  # Higher = lower FPS = slower particles

        self._particles = []
        self._logo = QtGui.QPixmap(logo_path) if logo_path and Path(logo_path).exists() else QtGui.QPixmap()

        # Buttons on top of the painted canvas
        self._btn_run = QtWidgets.QPushButton("▶ Run")
        self._btn_settings = QtWidgets.QPushButton("↓ Load Model")
        self._btn_quit = QtWidgets.QPushButton("⏻ Quit")

        # Style the buttons a bit
        for b in (self._btn_run, self._btn_settings, self._btn_quit):
            b.setCursor(QtCore.Qt.PointingHandCursor)
            b.setMinimumWidth(150)
        self._btn_run.setStyleSheet("""
            QPushButton {
                /* shape & layout */
                border-radius: 10px;
                padding: 10px 2px;
                font-size: 20px;
                font-weight: 600;

                /* filled gradient */
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 rgba(45, 140, 255, 0.7), stop:1 rgba(31, 111, 216, 0.7));
                color: white;
                border: 2px solid transparent;

                /* icon spacing & size (if you add an icon later) */
                qproperty-iconSize: 18px 18px;
            }

            /* hover: a touch brighter */
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #4B9FFF, stop:1 #2D8CFF);
            }

            /* pressed: slightly darker & “pressed-in” feel */
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #1E66C1, stop:1 #16539E);
                padding-top: 15px;      /* nudge content down by 1px */
                padding-bottom: 13px;
            }

            /* keyboard focus ring (accessible) */
            QPushButton:focus {
                outline: none;
                border: none;     /* light blue ring */
            }
            """)

        self._btn_settings.setStyleSheet("""
            QPushButton {
                border-radius: 10px;
                padding: 10px 2px;
                font-size: 20px;
                font-weight: 600;

                /* outline variant */
                background: transparent;
                color: white;
                border: 2px solid #5BA34E;

                qproperty-iconSize: 18px 18px;
            }

            /* hover: faint fill */
            QPushButton:hover {
                background: rgba(91, 163, 78, 0.50);
            }

            /* pressed: a bit darker border & stronger fill */
            QPushButton:pressed {
                background: rgba(91, 163, 78, 0.18);
                border-color: #4D8C40;
                padding-top: 15px;
                padding-bottom: 13px;
            }

            /* focus ring */
            QPushButton:focus {
                outline: none;
                border-color: none;     /* accessible ring */
            }
            """)

        self._btn_quit.setStyleSheet("""
            QPushButton {
                border-radius: 10px;
                padding: 10px 2px;
                font-size: 20px;
                font-weight: 600;

                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 rgba(195, 74, 71, 0.7), stop:1 rgba(159, 59, 56, 0.7));
                color: white;
                border: 2px solid transparent;

                qproperty-iconSize: 18px 18px;
            }

            /* hover */
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #D4605C, stop:1 #B04A46);
            }

            /* pressed */
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #9A3A37, stop:1 #7E2F2C);
                padding-top: 15px;
                padding-bottom: 13px;
            }

            /* focus ring */
            QPushButton:focus {
                outline: none;
                border: none;   /* soft red ring */
            }
            """)


        # Layout
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(250, 10, 250, 150)
        outer.addStretch(1)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._btn_run)
        row.addSpacing(16)
        row.addWidget(self._btn_settings)
        row.addSpacing(16)
        row.addWidget(self._btn_quit)
        row.addStretch(1)

        outer.addLayout(row)

        # Connect
        self._btn_run.clicked.connect(self.startRequested)
        self._btn_settings.clicked.connect(self.settingsRequested)
        self._btn_quit.clicked.connect(self.quitRequested)

    def showEvent(self, e):
        super().showEvent(e)
        self._reseed_particles()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._reseed_particles()

    def _reseed_particles(self):
        # Keep a moderate amount for performance
        count = 500
        rect = QtCore.QRectF(0, 0, self.width(), self.height())
        self._particles = [Particle(rect) for _ in range(count)]

    def _tick(self):
        for p in self._particles:
            p.update()
        self.update()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        self._mouse = QtCore.QPointF(e.position())
        super().mouseMoveEvent(e)

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHints(
            QtGui.QPainter.Antialiasing |
            QtGui.QPainter.TextAntialiasing |
            QtGui.QPainter.SmoothPixmapTransform
        )

        # Background gradient
        grad = QtGui.QLinearGradient(0, 0, 0, self.height())

        black_col = QtGui.QColor(15, 15, 15)
        gray_col = QtGui.QColor(35, 35, 35)

        grad.setColorAt(0.0, black_col)
        grad.setColorAt(0.75, black_col)
        grad.setColorAt(1.0, gray_col)
        p.fillRect(self.rect(), QtGui.QBrush(grad))

        # Draw particles
        for part in self._particles:
            part.draw(p, self._mouse)

        # Centered logo
        if not self._logo.isNull():
            scale = 1.0
            lw, lh = self._logo.width(), self._logo.height()
            max_w = int(self.width() * 0.8)
            if lw > max_w:
                scale = max_w / lw
            w = int(lw * scale)
            h = int(lh * scale)

            # cache the scaled pixmap so it's not resampled every paint
            if getattr(self, "_scaled_logo_size", None) != (w, h):
                self._scaled_logo = self._logo.scaled(
                    w, h,
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )
                self._scaled_logo_size = (w, h)

            x = (self.width() - w) // 2
            y = (self.height() - h) // 2 - 100
            p.drawPixmap(x, y, self._scaled_logo)


        # title = "Auto QUantification of Images Learning Algorithm"
        # font = QtGui.QFont(self.font())
        # font.setPointSize(18)
        # font.setBold(True)
        # p.setFont(font)
        # p.setPen(QtGui.QColor(220, 220, 220))
        # metrics = QtGui.QFontMetrics(font)
        # tw = metrics.horizontalAdvance(title)
        # p.drawText((self.width()-tw)//2, (self.height()//2)+50, title)

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
