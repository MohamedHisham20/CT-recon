import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Apply dark theme to match the UI
matplotlib.rcParams.update({
    "axes.facecolor": "#121929",        # Darker inside charts
    "figure.facecolor": "#1B202D",       # Match main body
    "axes.edgecolor": "#3B4252",
    "axes.labelcolor": "#A0AEC0",
    "text.color": "#A0AEC0",
    "xtick.color": "#A0AEC0",
    "ytick.color": "#A0AEC0",
    "grid.color": "#2D3748",
    "figure.autolayout": True,
    "lines.linewidth": 1.5,
    "axes.prop_cycle": matplotlib.cycler(color=["#63B3ED", "#F56565", "#48BB78", "#ED8936"])
})

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
