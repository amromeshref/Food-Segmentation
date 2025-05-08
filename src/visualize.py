import cv2
import json
import numpy as np

VAL_ANN_DATA_PATH = "../data/external/public_validation_set_2.0/annotations.json"
TRAIN_ANN_DATA_PATH = "../data/external/public_training_set_release_2.0/annotations.json"
ANN_DATA = TRAIN_ANN_DATA_PATH

class Visualizer:
    def __init__(self):
        with open(ANN_DATA) as f:
            self.ann_data = json.load(f)
        with open("../data/external/food_classes_with_original_id.json") as f:
            self.food_classes = json.load(f)
        
    def visualize_bbox(self, image_id: int = -1):
        """
        Visualize a random annotation or a specific annotation with the given image_id
        Args:
            image_id (int): The image_id of the annotation to visualize. If -1, a random annotation will be visualized
        Returns:
            None
            Displays the image with the bounding box and the food class
        """

        # Choose a random annotation and visualize it
        if image_id == -1:
            random_index = np.random.randint(0, len(self.ann_data["annotations"]))
            ann = self.ann_data["annotations"][random_index]
            image_id = ann["image_id"]
            image_name = str(image_id)

            while(len(image_name) < 6):
                image_name = "0" + image_name
            
            if ANN_DATA == VAL_ANN_DATA_PATH:
                image_path = f"../data/external/public_validation_set_2.0/images/{image_name}.jpg"
            else:
                image_path = f"../data/external/public_training_set_release_2.0/images/{image_name}.jpg"
            
        else:
            image_name = str(image_id)
            while(len(image_name) < 6):
                image_name = "0" + image_name

            if ANN_DATA == VAL_ANN_DATA_PATH:
                image_path = f"../data/external/public_validation_set_2.0/images/{image_name}.jpg"
            else:
                image_path = f"../data/external/public_training_set_release_2.0/images/{image_name}.jpg"
        
        image = cv2.imread(image_path)
        original_image = image.copy()  # Keep a copy of the original image for later use

        detected_classes = set()  # Set to keep track of the unique classes detected

        for ann in self.ann_data["annotations"]:
            if ann["image_id"] == image_id:
                bbox = ann["bbox"]
                x, y, w, h = bbox
                x, y, w, h = int(x), int(y), int(w), int(h)
                category_id = ann["category_id"]

                food_class_name = ""
                for food_class in self.food_classes:
                    if self.food_classes[food_class] == category_id:
                        food_class_name = food_class
                        break

                # Draw bounding box and put class name
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, food_class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Add the detected class to the set
                detected_classes.add(food_class_name)

        # Print the detected classes
        print(f"Classes found in bounding boxes: {', '.join(detected_classes)}")

        self.show_image(image, original_image)

    def visualize_segmentation(self, image_id: int = -1, show_text: bool = False):
        """
        Visualize a random annotation or a specific annotation with the given image_id
        Args:
            image_id (int): The image_id of the annotation to visualize. If -1, a random annotation will be visualized
            show_text (bool): If True, show the food class name on the image
        Returns:
            None
            Displays the image with the segmentation and the food class
        """
        # Choose a random annotation and visualize it
        if image_id == -1:
            random_index = np.random.randint(0, len(self.ann_data["annotations"]))
            ann = self.ann_data["annotations"][random_index]
            image_id = ann["image_id"]
            image_name = str(image_id)

            while(len(image_name) < 6):
                image_name = "0" + image_name
            
            if ANN_DATA == VAL_ANN_DATA_PATH:
                image_path = f"../data/external/public_validation_set_2.0/images/{image_name}.jpg"
            else:
                image_path = f"../data/external/public_training_set_release_2.0/images/{image_name}.jpg"
            
        else:
            image_name = str(image_id)
            while(len(image_name) < 6):
                image_name = "0" + image_name

            if ANN_DATA == VAL_ANN_DATA_PATH:
                image_path = f"../data/external/public_validation_set_2.0/images/{image_name}.jpg"
            else:
                image_path = f"../data/external/public_training_set_release_2.0/images/{image_name}.jpg"
        
        image = cv2.imread(image_path)
        original_image = image.copy()  # Keep a copy of the original image for later use
        mask = np.zeros_like(image)

        detected_classes = set()  # Set to keep track of the unique classes detected

        for ann in self.ann_data["annotations"]:
            if ann["image_id"] == image_id:
                segmentation = ann["segmentation"]
                category_id = ann["category_id"]

                food_class_name = ""
                for food_class in self.food_classes:
                    if self.food_classes[food_class] == category_id:
                        food_class_name = food_class
                        break

                for segment in segmentation:
                    points = np.array(segment, dtype=np.int32).reshape(-1, 2)  # Reshape to Nx2
                    cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)
                    cv2.fillPoly(mask, [points], color=(0, 255, 0))  # Fill mask with green

                    if show_text:
                        # Calculate the centroid (center) of the polygon
                        M = cv2.moments(points)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                        else:
                            cX, cY = 0, 0

                        # Put text near the centroid of the shape
                        cv2.putText(image, food_class_name, (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

                # Add the detected class to the set
                detected_classes.add(food_class_name)

        # Print the detected classes
        print(f"Classes found in segmentation: {', '.join(detected_classes)}")

        # Show the image with the segmentation
        self.show_image(image, original_image)


    def show_image(self, image, original_image):
        cv2.imshow('image', image)
        cv2.imshow('original image', original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    visualizer = Visualizer()
    #visualizer.visualize_bbox(134535)
    visualizer.visualize_segmentation(136406)
