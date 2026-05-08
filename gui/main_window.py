from PyQt5.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget
from PyQt5.QtGui import QFont, QIcon
import qtawesome as qta
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from gui.tabs import TabScenario1, TabScenario2, TabScenario3

# A Modern Dark Theme QSS (Qt StyleSheet)
STYLESHEET = """
QMainWindow {
    background-color: #1B202D;
}
QWidget {
    background-color: #1B202D;
    color: #E2E8F0;
    font-family: 'Segoe UI', Arial, sans-serif;
}
/* Side Panel Styling */
#SidePanel {
    background-color: #121929;
    border: 1px solid #2D3748;
    border-radius: 6px;
    margin: 5px;
    padding: 10px;
}
#SidePanel QLabel {
    background-color: transparent;
    border: none;
    font-size: 13px;
}
/* Tab Styling */
QTabWidget::pane {
    border: 1px solid #2D3748;
    background: #1B202D;
    border-radius: 6px;
}
QTabBar::tab {
    background: #121929;
    color: #A0AEC0;
    border: 1px solid #2D3748;
    border-bottom-color: #2D3748; 
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 20px;
    margin-right: 2px;
    font-weight: bold;
    font-size: 13px;
}
QTabBar::tab:selected {
    background: #2B6CB0;
    color: #FFFFFF;
}
QTabBar::tab:hover {
    background: #2C5282;
}

/* Button Styling */
QPushButton {
    background-color: #3182CE;
    color: white;
    border-radius: 5px;
    padding: 8px;
    margin-top: 10px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #2B6CB0;
}
QPushButton:pressed {
    background-color: #2C5282;
}
QPushButton:disabled {
    background-color: #4A5568;
    color: #A0AEC0;
}

/* Checkbox */
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #4A5568;
    background-color: #2D3748;
}
QCheckBox::indicator:checked {
    background-color: #3182CE;
    border: 1px solid #3182CE;
}

/* Scroll Area */
QScrollArea {
    border: 1px solid #2D3748;
    background-color: #1B202D;
    border-radius: 6px;
}
QScrollBar:vertical {
    border: none;
    background: #121929;
    width: 12px;
    border-radius: 6px;
}
QScrollBar::handle:vertical {
    background: #4A5568;
    min-height: 20px;
    border-radius: 6px;
}
QScrollBar::handle:vertical:hover {
    background: #718096;
}
QProgressBar {
    border: 1px solid #4A5568;
    border-radius: 4px;
    text-align: center;
    background-color: #2D3748;
    color: white;
}
QProgressBar::chunk {
    background-color: #3182CE;
}
"""

class CTMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sampling Studio - CT Edition")
        self.resize(1300, 850)
        self.setStyleSheet(STYLESHEET)
        
        # Initialize Phantom
        raw_phantom = shepp_logan_phantom()
        self.phantom = resize(raw_phantom, (256, 256), mode='reflect', anti_aliasing=True)
        
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Tabs Setup
        self.tabs = QTabWidget()
        self.tab1 = TabScenario1(self.phantom)
        self.tab2 = TabScenario2(self.phantom)
        self.tab3 = TabScenario3(self.phantom)
        
        # Add Icons using qtawesome (cloud, slider, eye, etc.)
        icon1 = qta.icon('fa5s.chart-line', color='white')
        icon2 = qta.icon('fa5s.eye', color='white')
        icon3 = qta.icon('fa5s.cogs', color='white')
        
        self.tabs.addTab(self.tab1, icon1, " Dose vs Noise (NPS)")
        self.tabs.addTab(self.tab2, icon2, " FBP vs TV Denoising (CNR)")
        self.tabs.addTab(self.tab3, icon3, " Sparse Angular Sampling (ART)")
        
        main_layout.addWidget(self.tabs)
