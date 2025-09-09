# Bhuvan Geo-AI Detector: Automated Ship and Airplane Detection

[cite_start]This project integrates the Bhuvan WMS (Web Map Service) with a YOLOv11 object detection model to automatically detect ships and airplanes in satellite imagery.

## Features

* [cite_start]**Automated Object Detection**: Utilizes a YOLOv11 model to detect ships and airplanes.
* [cite_start]**Bhuvan WMS Integration**: Fetches satellite imagery directly from Bhuvan WMS for real-time analysis.
* [cite_start]**Interactive Web Interface**: A Streamlit application (`app.py`) provides a user-friendly interface to perform detections on a map.
* [cite_start]**Geospatial Data Export**: Detections can be exported as GeoJSON and Shapefiles for use in GIS software.

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/Bhuvan-Geo-AI-Detector.git](https://github.com/your-username/Bhuvan-Geo-AI-Detector.git)
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit application:

```bash
streamlit run app.py
```

## Project Overview

This project is divided into several key stages, each detailed in the Jupyter notebooks in the `notebooks/` directory:

1.  [cite_start]**Data Augmentation**: `1_Data_Augmentation.ipynb` - Shows the process of augmenting the training data for the YOLOv11 model using various techniques.
2.  [cite_start]**Model Training**: `2_Model_Training.ipynb` - Details the training process of the YOLOv11 model on the augmented dataset.
3.  [cite_start]**WMS Integration and Inference**: `3_WMS_Integration_and_Inference.ipynb` - Explains how to programmatically access Bhuvan's WMS and run inference with the trained model.
4.  [cite_start]**Region of Interest (ROI) Extraction**: `4_ROI_Extraction.ipynb` - Demonstrates how to extract specific regions of interest from the Bhuvan Geo-Portal.

[cite_start]For a detailed explanation of the project, please see the full internship report in the `report/` directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.