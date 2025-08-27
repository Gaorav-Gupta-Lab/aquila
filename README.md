# AQUILA

AQUILA is a Python application with a Qt-based GUI for microscopy foci analysis and visualization.

## Features

- **Pipeline Functionalities:**
	- Specify input directory for reading files  
	- Specify output directory for image annotations and summary dataframe/plots

- **Utilities:**
	- Robust image reading for different file types.
	- Difference of Gaussians (DoG) image blurring for robust foci detectiion.
	- Nuclear segmentation.
	- Output plot for integrated data analysis.

- **Adjustable Parameters:**	
	- Foci Detection
		- DoG sigma values: Controls image blurring and foci calling (Sigma A subtracted from Sigma B).
		- Prominence: Sets minimum intensity to consider calling a focus.
	- Nuclear Segmentation
		- Min and Max Area: Sets pixel size of nuclei to exclude undesired nuclei or clumps.
		- DAPI Foreground: Aesthetic preference for nuclear segmentation (light nuclei on dark background or dark nuclei on light background)
		- Sigma Blur: Blurs nuclei to make threhsolding more stable and ensure proper nucleus segmentation.
		- Seed radius: Controls nuclear erosion and segmentation during thresholding.
	- Advanced
		- Maxima Sigma Smoothing: Applies blur to image after DoG for robust foci detection.
		- Minimum Distance Between Maxima: Sets minimum distance between detected foci to avoid counting the same focus twice.
		- Extensions: Specifies what image extensions should be considered when extracting images from the directory.


## Files

- **aquila_main.py** — The main entry point for the GUI application.  
	- Launches the main window (`AquilaWindow`) and menu screen.
	- Handles user interaction via Qt widgets.

- **aquila_utils.py** — Utility functions and classes for image processing, file handling, and plotting.  
	- Robust image reading (TIFF, PNG, JPEG, etc.).
	- Helper functions.
	- Uses NumPy, pandas, OpenCV, scipy and scikit-image libraries for analysis. Matplotlib and seaborn are used for plotting functions.

- **AquilaWindow.py** — Harbors AquilaWindow class

- **MenuScreen.py** — Harbors MenuScreen class

## Requirements

- Python 3.12+
- PySide6
- PyObjC (Optional, for loading icon on Mac only)
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

1. Run the main application from the downloaded directory with `AquilaWindow.py`, `MenuScreen.py`, `aquila_main.py` and `aquila_utils.py`:

```
sh python aquila_main.py
```

2. The GUI will launch with a menu and options to run analysis.

## License

See [LICENSE](LICENSE) for details.

## Contributing
If you would like to contribute to this project, feel free to open an issue or submit a pull request. All suggestions and improvements are welcome.

## Contact
For any inquiries, you can reach out to the developer at: pguerra@unc.edu
