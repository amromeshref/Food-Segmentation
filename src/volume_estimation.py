import pyzed.sl as sl
import cv2
import numpy as np
from scipy.spatial import ConvexHull
from segmentation_model import SegmentationModel


class FoodVolumeEstimator(SegmentationModel):
    def __init__(self):
        super().__init__()
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.camera_fps = 30

        if self.zed.open(self.init_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to open ZED camera")

        self.left_image = sl.Mat()
        self.depth_map = sl.Mat()
        cv2.namedWindow("ZED2 Food Volume Estimation", cv2.WINDOW_NORMAL)

    def polygon_to_mask(self, polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Convert polygon points to binary mask."""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        return mask

    def compute_volume_from_mask(self, mask: np.ndarray, depth_map: sl.Mat) -> float:
        """Estimate volume using 3D points inside a mask and ConvexHull."""
        height, width = mask.shape
        points_3d = []

        for y in range(height):
            for x in range(width):
                if mask[y, x]:
                    err, point = depth_map.get_value(x, y)
                    if err == sl.ERROR_CODE.SUCCESS and not np.isnan(point[0]):
                        points_3d.append([point[0], point[1], point[2]])

        if len(points_3d) < 4:
            return 0.0

        try:
            hull = ConvexHull(points_3d)
            return hull.volume
        except Exception:
            return 0.0

    def run(self):
        """Main loop to capture images, perform segmentation, and estimate volumes."""
        while True:
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.left_image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.depth_map, sl.MEASURE.XYZ)
                image_np = self.left_image.get_data()

                # Run segmentation
                results = self.segment(image_np)
                polygons = self.get_object_polygon_points(results)
                object_names = self.get_object_names(results)

                for i, polygon in enumerate(polygons):
                    mask = self.polygon_to_mask(polygon, image_np.shape)
                    volume = self.compute_volume_from_mask(mask, self.depth_map)

                    name = object_names[i] if i < len(object_names) else "Object"
                    label = f"{name}: {volume:.4f} m³"

                    # Draw polygon and label
                    cv2.polylines(image_np, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.putText(image_np, label, tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 255), 2)

                cv2.imshow("ZED2 Food Volume Estimation", image_np)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        """Release resources."""
        self.zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    estimator = FoodVolumeEstimator()
    estimator.run()
