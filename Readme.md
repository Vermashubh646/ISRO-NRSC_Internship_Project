# Bhuvan Geo-AI Detector: Automated Ship and Airplane Detection
<img width="1920" height="1080" alt="screenshot_bhuvan_geo_ai_detector (1)" src="https://github.com/user-attachments/assets/cbb88694-2fae-43c8-96f6-fafa28984f38" />


This project integrates the Bhuvan WMS (Web Map Service) with a YOLOv11 object detection model to automatically detect ships and airplanes in satellite imagery.

## Features

* **Automated Object Detection**: Utilizes a YOLOv11 model to detect ships and airplanes.
* **Bhuvan WMS Integration**: Fetches satellite imagery directly from Bhuvan WMS for real-time analysis.
* **Interactive Web Interface**: A Streamlit application (`app.py`) provides a user-friendly interface to perform detections on a map.
* **Geospatial Data Export**: Detections can be exported as GeoJSON and Shapefiles for use in GIS software.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Vermashubh646/ISRO-NRSC_Internship_Project.git
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit application:

```bash
streamlit run Streamlit/app.py
```

## Project Overview

This project is divided into several key stages, each detailed in the Jupyter notebooks in the `notebooks/` directory:

1.  `Extraction_roi_maps.ipynb` & `Integrating_WMS.ipynb` - Demonstrates how to extract satellite images and  specific regions of interest from the Bhuvan Geo-Portal.
2.  `augmentation.ipynb` - Shows the process of augmentation the training data for the YOLOv11 model using various techniques.
3.  `Model_Training.ipynb` - Details of the training process of the YOLOv11 model on the augmented dataset.
4.  `Inference_Pipeline.ipynb` - Contains functions to programmatically access Bhuvan's WMS and run inference with the trained model.

