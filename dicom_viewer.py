import sys
import os
import pydicom
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QListWidget, 
                             QLabel, QSplitter, QSlider)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=6, dpi=100):
        fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.axis('off')
        super(MplCanvas, self).__init__(fig)

class ExternalDicomViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("External DICOM Viewer")
        self.resize(1000, 700)
        self.setStyleSheet("""
            QMainWindow { background-color: #1B202D; }
            QWidget { background-color: #1B202D; color: #E2E8F0; font-family: 'Segoe UI', Arial, sans-serif; }
            QListWidget { background: #121929; border: 1px solid #2D3748; border-radius: 6px; padding: 5px; }
            QListWidget::item:selected { background: #3182CE; color: white; }
            QPushButton { background-color: #3182CE; color: white; border-radius: 5px; padding: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #2B6CB0; }
        """)
        
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Splitter to allow resizing side panel vs image
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left Panel (Controls + List)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        self.btn_load_dir = QPushButton("Load DICOM Folder")
        self.btn_load_dir.clicked.connect(self.load_directory)
        left_layout.addWidget(self.btn_load_dir)
        
        self.list_widget = QListWidget()
        self.list_widget.itemSelectionChanged.connect(self.display_dicom)
        left_layout.addWidget(self.list_widget)
        
        # Right Panel (Info + Viewer)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        self.lbl_info = QLabel("Select a DICOM folder to start.")
        self.lbl_info.setStyleSheet("font-size: 14px; font-weight: bold; color: #A0AEC0;")
        right_layout.addWidget(self.lbl_info)
        
        self.canvas = MplCanvas(self, width=6, height=6, dpi=100)
        right_layout.addWidget(self.canvas)
        
        # Contrast / Windowing Controls
        slider_layout = QVBoxLayout()
        slider_layout.setContentsMargins(0, 10, 0, 0)
        
        center_layout = QHBoxLayout()
        self.lbl_center = QLabel("Window Center (Level): 40")
        self.slider_center = QSlider(Qt.Horizontal)
        self.slider_center.setRange(-1000, 3000)
        self.slider_center.setValue(40)
        self.slider_center.valueChanged.connect(self.update_image)
        center_layout.addWidget(self.lbl_center)
        center_layout.addWidget(self.slider_center)
        
        width_layout = QHBoxLayout()
        self.lbl_width = QLabel("Window Width: 400")
        self.slider_width = QSlider(Qt.Horizontal)
        self.slider_width.setRange(1, 4000)
        self.slider_width.setValue(400)
        self.slider_width.valueChanged.connect(self.update_image)
        width_layout.addWidget(self.lbl_width)
        width_layout.addWidget(self.slider_width)
        
        slider_layout.addLayout(center_layout)
        slider_layout.addLayout(width_layout)
        right_layout.addLayout(slider_layout)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700])
        
        self.current_folder = ""
        self.current_hu_image = None

    def load_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Directory")
        if folder:
            self.current_folder = folder
            self.list_widget.clear()
            
            # Find all files, not just .dcm
            files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            
            # Try to sort numerically if possible (e.g., 1.dcm, 2.dcm or 1, 2, 3)
            import re
            files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
            
            self.list_widget.addItems(files)
            if files:
                self.lbl_info.setText(f"Loaded {len(files)} slices from: {os.path.basename(folder)}")
            else:
                self.lbl_info.setText("No .dcm files found in the selected folder.")

    def display_dicom(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            return
        
        filename = selected_items[0].text()
        filepath = os.path.join(self.current_folder, filename)
        self.lbl_info.setText(f"Loading: {filename}...")
        
        try:
            ds = pydicom.dcmread(filepath)
            image = ds.pixel_array
            
            # Apply rescale to get True HU values
            intercept = getattr(ds, 'RescaleIntercept', 0)
            slope = getattr(ds, 'RescaleSlope', 1)
            self.current_hu_image = image * slope + intercept
            
            # Auto-detect DICOM window if available, else keep previous slider values
            if 'WindowCenter' in ds and 'WindowWidth' in ds:
                wc = ds.WindowCenter
                ww = ds.WindowWidth
                if isinstance(wc, pydicom.multival.MultiValue): wc = wc[0]
                if isinstance(ww, pydicom.multival.MultiValue): ww = ww[0]
                
                self.slider_center.blockSignals(True)
                self.slider_center.setValue(int(wc))
                self.slider_center.blockSignals(False)
                
                self.slider_width.blockSignals(True)
                self.slider_width.setValue(int(ww))
                self.slider_width.blockSignals(False)
            
            self.lbl_info.setText(f"File: {filename} | Shape: {image.shape}")
            self.update_image()
            
        except Exception as e:
            self.lbl_info.setText(f"Error loading {filename}")
            print(f"Failed to read DICOM: {e}")

    def update_image(self):
        if self.current_hu_image is None:
            return
            
        center = self.slider_center.value()
        width = self.slider_width.value()
        
        self.lbl_center.setText(f"Window Center (Level): {center}")
        self.lbl_width.setText(f"Window Width: {width}")
        
        lower_bound = center - (width / 2.0)
        upper_bound = center + (width / 2.0)
        
        windowed_image = np.clip(self.current_hu_image, lower_bound, upper_bound)
        
        # Render to canvas
        self.canvas.axes.clear()
        self.canvas.axes.imshow(windowed_image, cmap='gray', vmin=lower_bound, vmax=upper_bound)
        self.canvas.axes.axis('off')
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ExternalDicomViewer()
    viewer.show()
    sys.exit(app.exec_())
