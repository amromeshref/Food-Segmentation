from ultralytics import YOLO
import cv2
import numpy as np


WEIGHTS_PATH = "../weights/last.pt"

class SegmentationModel:
    def __init__(self):
        self.model = YOLO(WEIGHTS_PATH)

    def segment(self, image: np.ndarray):
        """
        Perform segmentation on the input image.

        Args:
            image (np.ndarray): The input image in BGR format.

        Returns:
            results (object): The segmentation results.
        """
        # Convert the image to RGB format.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform segmentation on the image.
        results = self.model.predict(image_rgb, conf=0.5, show=True, save=False)
        return results[0]

    def get_object_polygon_points(self, results) -> list[np.ndarray]:
        """
        Get the object polygon points as integers which are the masks of the detected objects in the image.
        They are x, y coordinates of the polygon points of the detected objects.
        Args:
            results: results of the segmentation model.
        Returns:
            list[np.ndarray]: list of polygons of detected objects.
        """
        masks = results.masks

        # Return an empty list if no objects are detected.
        if masks is None:
            return []

        # get the x, y coordinates of the polygon points of the detected objects.
        polygon_points = masks.xy

        # convert the x, y coordinates to integers.
        polygon_points_int = [array.astype(int) for array in polygon_points]

        return polygon_points_int
    
    def color_polygon_region(self, image, results, mask_color=(255, 0, 0)) -> np.ndarray:
        """
        Color the region of the detected object in the image.
        Args:
            image (np.ndarray - BGR format): image to color the detected object region in.
            results: results of the segmentation model.
            mask_color (tuple): color to use for coloring the detected object region.
        Returns:
            np.ndarray (BGR format): image with the detected object region colored.
        """
        # Get the polygon points of the detected objects.
        polygon_points = self.get_object_polygon_points(results)

        # Return the original image if no objects are detected.
        if len(polygon_points) == 0:
            return image

        # Fill the detected objects region with the mask color.
        image_copy = image.copy()
        for polygon in polygon_points:
            cv2.fillPoly(image_copy, [polygon], mask_color)

        return image_copy
    
    def get_object_names(self, results) -> list[str]:
        """
        Get the names of the detected objects in the image.
        Args:
            results: results of the segmentation model.
        Returns:
            list[str]: list of names of the detected objects.
        """
        names = results.names
        return [names[int(cls)] for cls in results.boxes.cls]
    

if __name__ == "__main__":
    # Example usage
    model = SegmentationModel()
    image = cv2.imread("../images/024678.jpg")
    results = model.segment(image)
    colored_image = model.color_polygon_region(image, results)
    object_names = model.get_object_names(results)
    print("Detected objects:", object_names)
    cv2.imshow("Colored Image", colored_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()