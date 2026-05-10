from PyQt5.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QFileDialog, QSlider
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
import qtawesome as qta
import os
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from gui.tabs import TabScenario1, TabScenario2, TabScenario3
from core.data_loader import load_clinical_phantom

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
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_dicom_path = os.path.join(project_root, "sample_data", "I238.dcm")
        
        self.current_dicom_path = None
        if os.path.exists(default_dicom_path):
            self.phantom = load_clinical_phantom(default_dicom_path)
            self.current_dicom_path = default_dicom_path
            default_label = "Using: I238.dcm"
        else:
            raw_phantom = shepp_logan_phantom()
            self.phantom = resize(raw_phantom, (256, 256), mode='reflect', anti_aliasing=True)
            default_label = "Using: Shepp-Logan Phantom"
        
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Phantom Selection Layout
        phantom_layout = QHBoxLayout()
        self.btn_load_dicom = QPushButton("Load Clinical DICOM (.dcm)")
        self.btn_reset_phantom = QPushButton("Reset to Phantom")
        self.lbl_dicom_file = QLabel(default_label)
        
        self.btn_load_dicom.clicked.connect(self.load_dicom_phantom)
        self.btn_reset_phantom.clicked.connect(self.load_shepp_logan)
        
        # Contrast Sliders
        self.lbl_center = QLabel("Center: 47")
        self.slider_center = QSlider(Qt.Horizontal)
        self.slider_center.setRange(-1000, 3000)
        self.slider_center.setValue(47)
        self.slider_center.setFixedWidth(100)
        self.slider_center.sliderReleased.connect(self.update_contrast)
        
        self.lbl_width = QLabel("Width: 69")
        self.slider_width = QSlider(Qt.Horizontal)
        self.slider_width.setRange(1, 4000)
        self.slider_width.setValue(69)
        self.slider_width.setFixedWidth(100)
        self.slider_width.sliderReleased.connect(self.update_contrast)
        
        phantom_layout.addWidget(self.btn_load_dicom)
        phantom_layout.addWidget(self.btn_reset_phantom)
        phantom_layout.addWidget(self.lbl_dicom_file)
        phantom_layout.addStretch()
        phantom_layout.addWidget(self.lbl_center)
        phantom_layout.addWidget(self.slider_center)
        phantom_layout.addWidget(self.lbl_width)
        phantom_layout.addWidget(self.slider_width)
        main_layout.addLayout(phantom_layout)
        
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

    def load_shepp_logan(self):
        raw_phantom = shepp_logan_phantom()
        self.phantom = resize(raw_phantom, (256, 256), mode='reflect', anti_aliasing=True)
        self.current_dicom_path = None
        self.lbl_dicom_file.setText("Using: Shepp-Logan Phantom")
        self.update_tabs()

    def update_contrast(self):
        center = self.slider_center.value()
        width = self.slider_width.value()
        self.lbl_center.setText(f"Center: {center}")
        self.lbl_width.setText(f"Width: {width}")
        
        if self.current_dicom_path and os.path.exists(self.current_dicom_path):
            self.phantom = load_clinical_phantom(self.current_dicom_path, window_center=center, window_width=width)
            self.update_tabs()

    def update_tabs(self):
        self.tab1.phantom = self.phantom
        self.tab2.phantom = self.phantom
        self.tab3.phantom = self.phantom

    def load_dicom_phantom(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open DICOM File", "", "All Files (*);;DICOM Files (*.dcm)")
        if file_path:
            try:
                self.current_dicom_path = file_path
                center = self.slider_center.value()
                width = self.slider_width.value()
                
                self.phantom = load_clinical_phantom(file_path, window_center=center, window_width=width)
                filename = os.path.basename(file_path)
                self.lbl_dicom_file.setText(f"Using: {filename}")
                self.update_tabs()
            except Exception as e:
                self.lbl_dicom_file.setText(f"Error loading {os.path.basename(file_path)}")
                print(f"Failed to load DICOM: {e}")
