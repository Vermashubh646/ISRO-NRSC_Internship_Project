import streamlit as st
import math
import re
import requests
from PIL import Image
from io import BytesIO
import zipfile
from pyproj import Transformer, CRS
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import supervision as sv
import numpy as np
import os
import fiona
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as shapely_transform
import geojson
import random

# Model Caching
@st.cache_resource
def load_yolo_model():
    """Loads the YOLOv8 model from a local path."""
    model_path = "../models/All Results/yolov11s_75_1920_highres_new_hv/weights/best.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at '{model_path}'. Please update the path.")
        return None
    model = YOLO(model_path)
    return model

# Geospatial and Image Generation Functions 

def find_satellite_layers(base_url):
    """Sends a GetCapabilities request to find available high-resolution satellite layers."""
    params = {'service': 'WMS', 'version': '1.1.1', 'request': 'GetCapabilities'}
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        all_layers = [name.text for name in root.findall('.//Layer/Name')]
        satellite_layers = [name for name in all_layers if name and ('hrs' in name or 'sat' in name)]
        return satellite_layers if satellite_layers else []
    except requests.exceptions.RequestException as e:
        st.error(f"Could not get server capabilities: {e}")
        return None

def get_tile_info(lon, lat, zoom):
    """Calculates the exact tile index (x, y) and bounding box for a given lat/lon and zoom level."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    tile_x = int((lon + 180.0) / 360.0 * n)
    tile_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

    def tile_to_lon(x, z): return x / (2.0 ** z) * 360.0 - 180.0
    def tile_to_lat(y, z):
        n = math.pi - 2.0 * math.pi * y / (2.0 ** z)
        return math.degrees(math.atan(0.5 * (math.exp(n) - math.exp(-n))))

    lon_deg_nw, lat_deg_nw = tile_to_lon(tile_x, zoom), tile_to_lat(tile_y, zoom)
    lon_deg_se, lat_deg_se = tile_to_lon(tile_x + 1, zoom), tile_to_lat(tile_y + 1, zoom)

    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    min_x, min_y = transformer.transform(lon_deg_nw, lat_deg_se)
    max_x, max_y = transformer.transform(lon_deg_se, lat_deg_nw)
    
    return {'x': tile_x, 'y': tile_y, 'bbox': (min_x, min_y, max_x, max_y)}

def generate_stitched_map(lon, lat, zoom, dynamic_layer_name, target_width=1904, target_height=937):
    """Generates a stitched map and returns the image along with geospatial metadata."""
    st.info(f"Generating a {target_width}x{target_height} map for layer: `{dynamic_layer_name}`...")
    
    is_preview_map = "bhuvan_ocm_wbase" in dynamic_layer_name
    base_url = "https://bhuvan-ras1.nrsc.gov.in/tilecache/tilecache.py" if is_preview_map else "https://bhuvan-ras1.nrsc.gov.in/SatServices/service"

    tile_pixel_size = 256
    grid_width = math.ceil(target_width / tile_pixel_size)
    grid_height = math.ceil(target_height / tile_pixel_size) + 2
    stitched_width, stitched_height = grid_width * tile_pixel_size, grid_height * tile_pixel_size

    central_tile = get_tile_info(lon, lat, zoom)
    c_min_x, _, c_max_x, c_max_y = central_tile['bbox']
    tile_coord_width = c_max_x - c_min_x

    offset_x, offset_y = math.floor(grid_width / 2), math.floor(grid_height / 2)
    grid_top_left_min_x = c_min_x - (offset_x * tile_coord_width)
    grid_top_left_max_y = c_max_y + (offset_y * tile_coord_width)

    stitched_image = Image.new('RGB', (stitched_width, stitched_height))
    
    wms_params = {'service': 'WMS', 'version': '1.1.1', 'request': 'GetMap', 'layers': dynamic_layer_name, 'styles': '', 'srs': 'EPSG:3857', 'format': 'image/png', 'width': 256, 'height': 256}
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://bhuvan.nrsc.gov.in/'}

    progress_bar = st.progress(0, text="Stitching map tiles...")
    total_tiles = grid_height * grid_width

    for i, (row, col) in enumerate(np.ndindex((grid_height, grid_width))):
        min_x = grid_top_left_min_x + col * tile_coord_width
        max_y = grid_top_left_max_y - row * tile_coord_width
        current_params = wms_params.copy()
        current_params['bbox'] = f"{min_x},{max_y - tile_coord_width},{min_x + tile_coord_width},{max_y}"
        try:
            response = requests.get(base_url, params=current_params, headers=headers)
            response.raise_for_status()
            tile_image = Image.open(BytesIO(response.content))
            stitched_image.paste(tile_image, (col * tile_pixel_size, row * tile_pixel_size))
        except (requests.exceptions.RequestException, IOError) as e:
            st.warning(f"Error fetching tile ({col}, {row}): {e}")
        progress_bar.progress((i + 1) / total_tiles, text=f"Stitching map tiles... ({i+1}/{total_tiles})")

    progress_bar.empty()
    center_x, center_y = stitched_width / 2, stitched_height / 2
    crop_left = int(center_x - target_width / 2)
    crop_top = int(center_y - target_height / 2 + 241)
    final_image = stitched_image.crop((crop_left, crop_top, crop_left + target_width, crop_top + target_height))
    st.success("Map generated successfully!")

    geospatial_metadata = {
        "stitched_top_left_x": grid_top_left_min_x,
        "stitched_top_left_y": grid_top_left_max_y,
        "meters_per_pixel": tile_coord_width / tile_pixel_size,
        "crop_box": (crop_left, crop_top, crop_left + target_width, crop_top + target_height)
    }
    
    return final_image, geospatial_metadata

def get_cropped_image_geocoords(metadata):
    """Calculates the geographic coordinates of the final cropped image's corners."""
    top_left_x = metadata["stitched_top_left_x"] + metadata["crop_box"][0] * metadata["meters_per_pixel"]
    top_left_y = metadata["stitched_top_left_y"] - metadata["crop_box"][1] * metadata["meters_per_pixel"]
    
    bottom_right_x = metadata["stitched_top_left_x"] + metadata["crop_box"][2] * metadata["meters_per_pixel"]
    bottom_right_y = metadata["stitched_top_left_y"] - metadata["crop_box"][3] * metadata["meters_per_pixel"]
    
    return {
        "top_left_mercator": (top_left_x, top_left_y),
        "bottom_right_mercator": (bottom_right_x, bottom_right_y)
    }

def create_geospatial_outputs(detections, model, image_geocoords, image_width, image_height):
    """Converts pixel detections to geographic polygons and creates GeoJSON and Shapefile."""
    features = []

    # Mercator bounds
    x_min_merc, y_max_merc = image_geocoords["top_left_mercator"]
    x_max_merc, y_min_merc = image_geocoords["bottom_right_mercator"]

    x_scale = (x_max_merc - x_min_merc) / image_width
    y_scale = (y_min_merc - y_max_merc) / image_height

    for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
        class_id = detections.class_id[i]
        confidence = detections.confidence[i]

        # Coordinates in EPSG:3857 (Mercator)
        p1 = (x_min_merc + x1 * x_scale, y_max_merc + y1 * y_scale)
        p2 = (x_min_merc + x2 * x_scale, y_max_merc + y1 * y_scale)
        p3 = (x_min_merc + x2 * x_scale, y_max_merc + y2 * y_scale)
        p4 = (x_min_merc + x1 * x_scale, y_max_merc + y2 * y_scale)

        poly_mercator = Polygon([p1, p2, p3, p4, p1])

        # Transform to WGS84 (EPSG:4326)
        transformer_to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
        poly_wgs84 = shapely_transform(transformer_to_wgs84, poly_mercator)

        feature = geojson.Feature(
            geometry=poly_wgs84,
            properties={"class_name": model.names[class_id], "confidence": float(confidence)}
        )
        features.append(feature)

    # Create GeoJSON
    feature_collection = geojson.FeatureCollection(features)
    geojson_data = geojson.dumps(feature_collection, indent=2).encode('utf-8')

    # Create Shapefile
    output_dir = r"shps"
    p = random.randint(1000, 10000)
    os.makedirs(f'{output_dir}/{p}', exist_ok=True)

    shp_path = os.path.join(f'{output_dir}/{p}', "detections.shp")

    schema = {
        'geometry': 'Polygon',
        'properties': {'class_name': 'str', 'confidence': 'float'}
    }

    with fiona.open(
        shp_path,
        mode='w',
        driver='ESRI Shapefile',
        crs='EPSG:4326',
        schema=schema,
        encoding='utf-8'
    ) as shp:
        for feat in features:
            shp.write({
                'geometry': mapping(feat['geometry']),
                'properties': feat['properties']
            })

    # Zip shapefile components
    zip_path = os.path.join(f'{output_dir}/{p}', "detections.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
            fpath = os.path.join(f'{output_dir}/{p}', f"detections{ext}")
            if os.path.exists(fpath):
                zipf.write(fpath, arcname=os.path.basename(fpath))

    with open(zip_path, 'rb') as f:
        shp_bytes = f.read()

    return geojson_data, shp_bytes
	
def run_detection_and_create_legend(img, model,confid=0.1):
    """Runs YOLO, annotates image, and returns data for a legend with counts."""
    results = model.predict(img, conf=confid)
	
    class_thresholds = {
    0: 0.6,  # airplane 
    1: 0.5,  # ship 
    2: 0.25,  # small ship 
    }

    final_xyxy = []
    final_conf = []
    final_cls = []


    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Check if the detection meets the custom threshold for its class
            if confidence >= class_thresholds.get(class_id, 0.1): # .get is safer
                final_xyxy.append(box.xyxy[0].cpu().numpy())
                final_conf.append(confidence)
                final_cls.append(class_id)


    if not final_xyxy:
        # If no objects are found, create an empty Detections object
        detections = sv.Detections.empty()
    else:
        # If objects are found, create the Detections object as before
        detections = sv.Detections(
            xyxy=np.array(final_xyxy),
            confidence=np.array(final_conf),
            class_id=np.array(final_cls)
        )

    color_palette = sv.ColorPalette.DEFAULT
    box_annotator = sv.BoxAnnotator(color=color_palette)
    
    annotated_image = box_annotator.annotate(scene=img.copy(), detections=detections)

    legend_data = {}
    detected_class_ids, counts = np.unique(detections.class_id, return_counts=True)
    class_counts = dict(zip(detected_class_ids, counts))

    for class_id in detected_class_ids:
        class_name = model.names[class_id]
        color_hex = color_palette.by_idx(class_id).as_hex()
        legend_data[class_name] = {"color": color_hex, "count": class_counts[class_id]}
        
    return annotated_image, detections, legend_data

# Streamlit App Main

def main():
    st.set_page_config(page_title="Bhuvan Map Detector", layout="wide")
    st.title("Bhuvan Geo-AI Detector")
    st.write("Leveraging AI to identify, analyze, and export ships and airplanes from ISRO's Bhuvan imagery.")

    if "satellite_layers" not in st.session_state:
        st.session_state.satellite_layers = None

    # SIDEBAR CONTROLS
    st.sidebar.header("Controls")
    
    st.sidebar.subheader("Enter Coordinates")
    lat = st.sidebar.number_input("Latitude", value=13.201950, format="%.6f")
    lon = st.sidebar.number_input("Longitude", value=77.701164, format="%.6f")
    
    #zoom = 16
    zoom = st.sidebar.radio(
        "Choose Zoom",
        (16, 18)
    )
	

    image_type_choice = st.sidebar.radio(
        "Choose Image Type",
        ("High-Res Satellite", "Low-Res Preview Map")
    )
    
	
	
    selected_layer = None
    if image_type_choice == "High-Res Satellite":
        if st.sidebar.button("Find Available Satellite Layers"):
            with st.spinner("Searching for layers..."):
                found_layers = find_satellite_layers("https://bhuvan-ras1.nrsc.gov.in/SatServices/service")
                if found_layers:
                    found_layers.reverse()
                st.session_state.satellite_layers = found_layers
        
        if st.session_state.satellite_layers is not None:
            if st.session_state.satellite_layers:
                st.sidebar.success(f"Found {len(st.session_state.satellite_layers)} layers!")
                selected_layer = st.sidebar.selectbox("Select a Satellite Layer", st.session_state.satellite_layers)
            else:
                st.sidebar.warning("Search complete, but no satellite layers were found.")
    else:
        selected_layer = "bhuvan_ocm_wbase"
        st.session_state.satellite_layers = None

    st.sidebar.write("---")
    
    if st.sidebar.button("Generate Map & Perform Detection", type="primary"):
        if lat is None or lon is None:
            st.error("Invalid coordinates. Please enter a valid Latitude and Longitude.")
            st.stop()

        model = load_yolo_model()
        if model is None: st.stop()

        st.sidebar.success(f"Using Coords: Lat={lat:.4f}, Lon={lon:.4f}")

        if selected_layer:
            with st.spinner("Generating map... This might take a moment."):
                original_image, geo_meta = generate_stitched_map(lon, lat, zoom, selected_layer)
            
            with st.spinner("Running detection model..."):
                annotated_image, detections, legend_data = run_detection_and_create_legend(original_image, model)
            
            # Store all results in st.session_state
            st.session_state.original_image = original_image
            st.session_state.annotated_image = annotated_image
            st.session_state.detections = detections
            st.session_state.legend_data = legend_data
            st.session_state.geo_meta = geo_meta
            st.session_state.model = model
            st.session_state.lat = lat
            st.session_state.lon = lon
            st.session_state.results_ready = True

        else:
            st.error("Action required: Please select a map layer in the sidebar first.")
            st.session_state.results_ready = False

    # This block is OUTSIDE the button's 'if' statement, so it runs on every script rerun.
    if st.session_state.get('results_ready', False):
        # Display Images and Legend 
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Map Image")
            st.image(st.session_state.original_image, use_container_width=True)
        with col2:
            st.subheader("Object Detection Results")
            st.image(st.session_state.annotated_image, use_container_width=True)
            st.success(f"Found {len(st.session_state.detections)} total objects.")

            if st.session_state.legend_data:
                st.subheader("Legend")
                for name, data in st.session_state.legend_data.items():
                    st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 5px;"><div style="width: 20px; height: 20px; background-color: {data["color"]}; margin-right: 10px; border: 1px solid black;"></div><span>{name}: {data["count"]}</span></div>', unsafe_allow_html=True)
        
        # Prepare and Display Download Buttons 
        if len(st.session_state.detections) > 0:
            with st.spinner("Preparing files for download..."):
                # Prepare image bytes
                original_img_bytes = BytesIO()
                st.session_state.original_image.save(original_img_bytes, format="PNG")
                
                annotated_img_bytes = BytesIO()
                st.session_state.annotated_image.save(annotated_img_bytes, format="PNG")
                
                # Prepare geospatial file bytes
                img_coords = get_cropped_image_geocoords(st.session_state.geo_meta)
                geojson_bytes, shp_bytes = create_geospatial_outputs(
                    st.session_state.detections, 
                    st.session_state.model, 
                    img_coords, 
                    st.session_state.original_image.width, 
                    st.session_state.original_image.height
                )

            st.write("---")
            st.subheader("Download Outputs")
            dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)

            # Retrieve coordinates for filenames
            lat = st.session_state.lat
            lon = st.session_state.lon

            with dl_col1:
                st.download_button("Original Image", original_img_bytes.getvalue(), f"original_{selected_layer}_{lat:.4f}_{lon:.4f}.png", "image/png")
            with dl_col2:
                st.download_button("Annotated Image", annotated_img_bytes.getvalue(), f"annotated_{selected_layer}_{lat:.4f}_{lon:.4f}.png", "image/png")
            with dl_col3:
                st.download_button("GeoJSON", geojson_bytes, f"detections_{selected_layer}_{lat:.4f}_{lon:.4f}.geojson", "application/geo+json")
            with dl_col4:
                st.download_button("Shapefile (.zip)", shp_bytes, f"detections_shp_{selected_layer}_{lat:.4f}_{lon:.4f}.zip", "application/zip")
				
if __name__ == "__main__":
    main()
