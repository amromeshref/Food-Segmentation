import pyzed.sl as sl
import cv2
from calorie_estimation import CalorieEstimator
from volume_estimation import FoodVolumeEstimator 


class RealTimeCalorieEstimator(FoodVolumeEstimator):
    def __init__(self):
        super().__init__()
        self.calorie_estimator = CalorieEstimator()

    def run(self):
        """Override run loop to include calorie estimation."""
        while True:
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.left_image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.depth_map, sl.MEASURE.XYZ)
                image_np = self.left_image.get_data()

                results = self.segment(image_np)
                polygons = self.get_object_polygon_points(results)
                object_names = self.get_object_names(results)

                for i, polygon in enumerate(polygons):
                    mask = self.polygon_to_mask(polygon, image_np.shape)
                    volume = self.compute_volume_from_mask(mask, self.depth_map)

                    name = object_names[i] if i < len(object_names) else "Object"
                    calories = self.calorie_estimator.estimate(name, volume)

                    label = f"{name}: {volume:.4f} m³ | {calories:.0f} kcal"

                    cv2.polylines(image_np, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.putText(image_np, label, tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 255), 2)

                cv2.imshow("ZED2 Real-Time Calorie Estimation", image_np)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()
