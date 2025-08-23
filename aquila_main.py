import sys, random, math
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets

from aquila_utils import AquilaWindow


# ---------------------------------------- 
# Particle System for Background Animation 
# ----------------------------------------

class Particle:
    def __init__(self, rect: QtCore.QRectF):
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

    def draw(self, painter: QtGui.QPainter, mouse: QtCore.QPointF):
        # dot
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(255, 255, 255, 220))
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
        self._timer.start(16)  # ~60 FPS

        self._particles = []
        self._logo = QtGui.QPixmap(logo_path) if logo_path and Path(logo_path).exists() else QtGui.QPixmap()

        # Buttons on top of the painted canvas
        self._btn_run = QtWidgets.QPushButton("Run")
        self._btn_settings = QtWidgets.QPushButton("Settings")
        self._btn_quit = QtWidgets.QPushButton("Quit")

        # Style the buttons a bit
        for b in (self._btn_run, self._btn_settings, self._btn_quit):
            b.setCursor(QtCore.Qt.PointingHandCursor)
            b.setMinimumWidth(120)
        self._btn_run.setStyleSheet("QPushButton{background:#2d8cff;color:white;border-radius:6px;padding:8px 16px;} QPushButton:hover{background:#4b9fff;}")
        self._btn_quit.setStyleSheet("QPushButton{background:#b0413e;color:white;border-radius:6px;padding:8px 16px;} QPushButton:hover{background:#c75c58;}")

        # Layout
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.addStretch(1)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._btn_run)
        row.addSpacing(8)
        row.addWidget(self._btn_settings)
        row.addSpacing(8)
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
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        # Background gradient
        grad = QtGui.QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, QtGui.QColor(18, 18, 18))
        grad.setColorAt(1.0, QtGui.QColor(32, 32, 32))
        p.fillRect(self.rect(), QtGui.QBrush(grad))

        # Draw particles
        for part in self._particles:
            part.draw(p, self._mouse)

        # Centered logo
        if not self._logo.isNull():
            scale = 1.0
            lw, lh = self._logo.width(), self._logo.height()
            # optional: scale logo if window small
            max_w = int(self.width() * 0.45)
            if lw > max_w:
                scale = max_w / lw
            w = int(lw * scale)
            h = int(lh * scale)
            x = (self.width() - w) // 2
            y = (self.height() - h) // 2 - 100
            target = QtCore.QRect(x, y, w, h)
            p.setOpacity(0.95)
            p.drawPixmap(target, self._logo)
            p.setOpacity(1.0)

        # Title (optional)
        title = "AQUILA — Auto Quantification of Images Learning Algorithm"
        font = QtGui.QFont(self.font())
        font.setPointSize(18)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QtGui.QColor(220, 220, 220))
        metrics = QtGui.QFontMetrics(font)
        tw = metrics.horizontalAdvance(title)
        p.drawText((self.width()-tw)//2, (self.height()//2)+20, title)

# --------------
# Main App Class
# --------------

class App(QtWidgets.QMainWindow):
    def __init__(self, logo_path=""):
        super().__init__()
        self.setWindowTitle("AQUILA")
        self.resize(1080, 720)

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

    def _go_run(self):
        # Jump straight to the parameters window (or trigger a default run)
        self.stack.setCurrentWidget(self.params)

    def _go_settings(self):
        # If you separate a settings screen, push that here
        self.stack.setCurrentWidget(self.params)


def main():
    app = QtWidgets.QApplication(sys.argv)
    # Point to your PNG logo (recommended transparent background)
    logo = r"C:\Users\pguer\OneDrive - University of North Carolina at Chapel Hill\Desktop\prog\GuptaLab\AQUILA\scripts\assets\aquila_full_resized.png"
    win = App(logo_path=logo)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
