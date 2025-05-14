# Food Volume and Calorie Estimation System

This project is designed to estimate the volume and calories of food items in real-time using computer vision techniques, depth sensing, and machine learning models. It utilizes a ZED 2 camera for depth estimation, YOLO-based segmentation for object detection, and custom algorithms for volume and calorie estimation.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [How It Works](#how-it-works)
  - [Segmentation](#segmentation)
  - [Volume Estimation](#volume-estimation)
  - [Calorie Estimation](#calorie-estimation)
  - [Recipe Suggestion](#recipe-suggestion)

## Overview

This system uses a ZED 2 camera for real-time 3D depth sensing and applies object detection, volume estimation, and calorie estimation algorithms to determine the size and calorie content of food items. It also suggests recipes based on the detected food items. The components of the system are modular, allowing for easy integration or extension.

## Requirements

The following dependencies are required to run the project:

- **Python 3.8+**
- **OpenCV** (cv2) - For image manipulation and video capture
- **PyZED** - For interacting with ZED cameras
- **Ultralytics YOLO** - Pre-trained object detection model
- **Pandas** - For handling food recipes in CSV format
- **Scipy** - For computing Convex Hull to estimate volume
- **NumPy** - For numerical operations
- **JSON** - For parsing annotation data

You can install the dependencies using pip:


## How It Works

This system integrates multiple components, including food item segmentation, volume estimation, and calorie estimation. Below is a breakdown of how each of these components works:

### Segmentation

The system uses **YOLO (You Only Look Once)**, a state-of-the-art real-time object detection model, to perform segmentation on images or video frames. YOLO detects food items by identifying the objects in an image and returning their bounding boxes and associated classes.

- The model is trained on a variety of food items and detects objects in an image.
- Segmentation results provide the names of detected food items and the polygon coordinates of their boundaries.
- These polygons represent the contours of the detected objects, which are essential for subsequent volume estimation.

#### File: `src/segmentation_model.py`
- **YOLO** model is loaded and used to segment food items in an image.
- **`segment()`** method processes the image and provides the object names and bounding polygons.
- **`get_object_polygon_points()`** extracts polygon coordinates, representing object boundaries, and **`get_object_names()`** maps them to recognizable food items.

### Volume Estimation

Volume estimation is achieved by using depth information provided by the **ZED 2 camera**, which captures 3D spatial data. After segmenting the food items, their 2D polygons are converted into masks that represent the regions of interest. These masks are used to extract 3D depth points from the depth map provided by the ZED camera.

- The **ZED 2 camera** provides depth information in real-time, which is used to get the 3D coordinates (X, Y, Z) of the segmented object.
- **Convex Hull** algorithm is applied to the 3D points inside the mask to estimate the volume of the object.
- The resulting volume is reported in cubic meters (m³).

#### File: `src/volume_estimation.py`
- **`polygon_to_mask()`** converts the detected polygon points to a binary mask.
- **`compute_volume_from_mask()`** estimates the 3D volume of the food item by constructing a Convex Hull around the 3D points inside the mask.
- The system continuously captures frames from the ZED 2 camera, processes them, and estimates the volume of the detected food items.

### Calorie Estimation

Once the food items are detected and their volumes are estimated, the system calculates the calorie content of the food. This is done using a predefined mapping of food items to their respective calorie counts, considering the estimated volume.

- The system uses an internal mapping or a dataset that associates food item names with average calorie values.
- The **calorie estimation** formula typically involves multiplying the volume of the food by a caloric density factor (calories per cubic meter).
- The final calorie count is displayed alongside the volume of the detected food items.

#### File: `src/real_time_calorie_estimation.py`
- **`real_time_calorie_estimation.py`** integrates the segmentation, volume estimation, and calorie estimation processes, running them sequentially on the live feed from the ZED camera.
- It calculates the calorie count for each food item detected in real-time and displays the results on the screen.

### Recipe Suggestion

For each detected food item, the system can suggest a recipe based on the food item’s name. Recipes are stored in a CSV file, and the system looks up the corresponding recipe based on the detected food item.

- The **recipe suggestion** feature uses a CSV database of food items and their corresponding recipes.
- After detecting the food items, the system queries the recipe database to retrieve and display a recipe for each food item.
- If no recipe is found for a detected food item, the system will notify the user that no recipe is available.

#### File: `src/recipe_suggestion.py`
- **`get_recipe()`** retrieves a recipe from the CSV database for each detected food item.
- **`suggest_recipes_from_image()`** detects food items in an image and suggests recipes for each detected food item.

### Summary of the Workflow

1. **Image Capture**: The ZED 2 camera continuously captures images and depth maps.
2. **Food Item Detection**: YOLO model performs segmentation to detect food items in the images.
3. **Volume Estimation**: Depth information is used to calculate the 3D volume of each detected food item.
4. **Calorie Calculation**: Based on the volume and a predefined caloric density, the system estimates the calorie count.
5. **Recipe Suggestion**: For each detected food item, a recipe is suggested from a CSV database.

The system is designed to run in real-time, offering immediate feedback on food item volumes, calories, and recipe suggestions.



