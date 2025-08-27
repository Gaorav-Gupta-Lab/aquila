# AQUILA

AQUILA is a Python application with a Qt-based GUI for microscopy foci analysis and visualization.

## Features

- **Image Utilities:**  
	Functions for robust image reading, processing, and manipulation.


## Files

- **aquila_main.py**  
	The main entry point for the GUI application.  
	- Launches the main window (`AquilaWindow`) and menu screen.
	- Implements a custom animated particle background.
	- Handles user interaction via Qt widgets.

- **aquila_utils.py**  
	Utility functions and classes for image processing and file handling.  
	- Robust image reading (TIFF, PNG, JPEG, etc.).
	- Helper functions for name truncation and more.
	- Uses NumPy, pandas, OpenCV, scikit-image, and other scientific libraries.

## Requirements

- Python 3.12+
- PySide6
- numpy
- pandas
- tifffile
- opencv-python
- Pillow
- scikit-image
- scipy
- matplotlib
- seaborn

Install dependencies with:
```sh
pip install -r requirements.txt
```

## Usage

1. Run the main application:
		```sh
		python aquila_main.py
		```

2. The GUI will launch with an animated menu and options to run analysis, load models, or quit.

## License

See [LICENSE](LICENSE) for details.
