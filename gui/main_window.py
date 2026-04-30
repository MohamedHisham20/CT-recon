from PyQt5.QtWidgets import QMainWindow, QTabWidget
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from gui.tabs import TabScenario1, TabScenario2, TabScenario3

class CTMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT Simulation & Reconstruction Dashboard")
        self.resize(1100, 800)
        
        # Initialize Phantom
        raw_phantom = shepp_logan_phantom()
        self.phantom = resize(raw_phantom, (256, 256), mode='reflect', anti_aliasing=True)
        
        # Tabs Setup
        self.tabs = QTabWidget()
        self.tab1 = TabScenario1(self.phantom)
        self.tab2 = TabScenario2(self.phantom)
        self.tab3 = TabScenario3(self.phantom)
        
        self.tabs.addTab(self.tab1, "1: Dose vs Noise (NPS)")
        self.tabs.addTab(self.tab2, "2: FBP vs TV Denoising (CNR)")
        self.tabs.addTab(self.tab3, "3: Sparse Angular Sampling (ART)")
        
        self.setCentralWidget(self.tabs)
