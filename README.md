# Upper-Body Clothing Detection with YOLOv8

Real-time upper-body clothing detection using [YOLOv8](https://docs.ultralytics.com/) and [OpenCV](https://opencv.org/). Detects six clothing categories from the [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) taxonomy via webcam or video input.

## Detected Categories

| Class | Label |
|-------|-------|
| 0 | Short Sleeve Top |
| 1 | Long Sleeve Top |
| 2 | Short Sleeve Outwear |
| 3 | Long Sleeve Outwear |
| 4 | Vest |
| 5 | Sling |

## Repository Structure

```
├── detect_top_clothes.py       # Main CLI script for real-time detection
├── detect_top_clothes.ipynb    # Jupyter notebook (local + Google Colab)
├── main.ipynb                  # Tie-detection notebook (bonus demo)
├── yolov8n.pt                  # Pre-trained YOLOv8 nano weights
└── README.md
```

## Requirements

- Python 3.8+
- [Ultralytics](https://pypi.org/project/ultralytics/) (YOLOv8)
- [OpenCV](https://pypi.org/project/opencv-python/)

## Installation

```bash
# Clone the repository
git clone https://github.com/AsadDeving/detect_topClothes.git
cd detect_topClothes

# Install dependencies
pip install ultralytics opencv-python
```

## Usage

### CLI Script

```bash
# Webcam (default)
python detect_top_clothes.py

# Video file
python detect_top_clothes.py --source video.mp4

# Custom confidence threshold and processing scale
python detect_top_clothes.py --source 0 --conf 0.6 --scale 0.75
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `0` | `0` for webcam, or path to a video file |
| `--conf` | `0.5` | Minimum detection confidence (0.0–1.0) |
| `--scale` | `0.5` | Frame scale for processing — lower values are faster |

Press **q** to quit the detection window.

### Jupyter Notebook

Open `detect_top_clothes.ipynb` in Jupyter or VS Code and run the cells sequentially. The notebook includes a dedicated **Google Colab** section that replaces `cv2.imshow()` with inline frame display.

### Tie Detection Demo

`main.ipynb` contains a separate demo that detects whether a person is wearing a tie. It uses a three-stage pipeline:

1. **Person detection** — locate people in the frame
2. **ROI extraction** — crop the chest-to-waist region
3. **Tie detection** — run YOLOv8 on the cropped region

## Configuration

### Using Custom Weights

Replace the model path at the top of `detect_top_clothes.py` (or in the notebook) to use your own DeepFashion2-trained weights:

```python
MODEL_PATH = "path/to/your/deepfashion2_weights.pt"
```

### Changing Target Classes

Edit the `TARGET_CLASSES` dictionary to detect different categories:

```python
TARGET_CLASSES = {
    0: "short sleeve top",
    1: "long sleeve top",
    2: "short sleeve outwear",
    3: "long sleeve outwear",
    4: "vest",
    5: "sling"
}
```

## How It Works

1. **Frame capture** — reads frames from a webcam or video file.
2. **Downscaling** — optionally resizes the frame for faster inference.
3. **YOLOv8 inference** — runs object detection on each frame.
4. **Class filtering** — keeps only the six target clothing classes.
5. **Visualization** — draws color-coded bounding boxes and confidence labels, then displays the annotated frame.
6. **Memory management** — periodically runs garbage collection to prevent memory leaks during long sessions.

## License

This project is provided as-is for educational and research purposes.
