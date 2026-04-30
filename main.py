import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import CTMainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = CTMainWindow()
    window.show()
    sys.exit(app.exec_())
