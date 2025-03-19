# Movement Detection with OpenCV

This project implements a simple movement detection system using OpenCV. It captures video from the webcam and processes the frames to detect any movement.

## Project Structure

```
opencv-movement-detection
├── src
│   ├── main.py          # Main entry point for the application
│   └── utils
│       └── __init__.py  # Utility functions for movement detection
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/opencv-movement-detection.git
   cd opencv-movement-detection
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the movement detection application, execute the following command:

```
python src/main.py
```

Make sure your webcam is connected and accessible.

## How It Works

The application initializes the webcam and continuously captures video frames. It processes each frame to detect movement by comparing it with the previous frame. If significant changes are detected, it highlights the areas of movement.

## Dependencies

This project requires the following Python packages:

- OpenCV
- NumPy

Make sure to install these packages using the `requirements.txt` file provided.