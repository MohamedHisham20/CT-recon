# CT Reconstruction Application (CT-recon)

A Python-based graphical desktop application for simulating Computed Tomography (CT) image acquisitions and performing various image reconstructions.

## Features
- Interactive Graphical User Interface (GUI) built with PyQt5.
- Simulation of CT measurements and physics (using Radon transforms).
- Multiple scenarios/tabs for testing different reconstruction algorithms (e.g., Filtered Back Projection, Algebraic Reconstruction Techniques).
- Quantitative metric evaluation including RMSE, PSNR, and CNR.

## Project Structure
- `main.py`: The main entry point to launch the application.
- `core/`: Contains the core mathematical and scientific logic.
  - `metrics.py`: Calculation of image quality metrics (PSNR, RMSE, CNR, etc.).
  - `physics.py`: Physics simulations (e.g., forward projection/measurements).
  - `reconstruction.py`: Implementations of different CT reconstruction algorithms.
- `gui/`: Contains all PyQt5 components and layouts.
  - `main_window.py`: The primary application window.
  - `tabs.py`: Specific UI tabs corresponding to different scenarios.
  - `workers.py`: Background worker threads to keep the UI responsive during heavy processing.
  - `components.py`: Custom GUI widgets, such as Matplotlib canvas integrations.

## Prerequisites
- Python 3.7 or higher installed on your system.
- It is recommended to use a virtual environment.

## Installation and Setup

1. **Clone or Download the Repository**
   Navigate to the directory where you want to place the project. If you're using git:
   ```bash
   git clone https://github.com/MohamedHisham20/CT-recon.git
   cd CT-recon
   ```

2. **(Optional) Create a Virtual Environment**
   It's best practice to isolate your project's dependencies:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS and Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   Install the required Python packages using the provided `requirements.txt` file (Make sure you are in the root directory of the project):
   ```bash
   pip install -r requirements.txt
   ```
   *The main dependencies include `numpy`, `scipy`, `PyQt5`, `scikit-image`, and `matplotlib`.*

## Running the Application

Once the dependencies are installed, you can launch the application by running the `main.py` script from the project's root directory:

```bash
python main.py
```

The GUI window should appear, granting you access to the CT reconstruction scenarios.
