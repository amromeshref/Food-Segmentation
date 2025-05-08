import pandas as pd

CALORIES_DATA_CSV_PATH = "../data/food_nutrition.csv"

class CalorieEstimator:
    def __init__(self, csv_path: str):
        """
        Initializes the calorie estimator using food density and calorie data.

        Args:
            csv_path (str): Path to the CSV file containing food data.
        """
        self.data = pd.read_csv(csv_path)
        self.data.set_index("Food Item", inplace=True)

    def estimate(self, food_name: str, volume_m3: float) -> float:
        """
        Estimate the total calories for a given food item and volume.

        Args:
            food_name (str): The name of the food item (must match CSV index).
            volume_m3 (float): The estimated volume in cubic meters.

        Returns:
            float: Estimated calories. Returns 0.0 if food not found or volume invalid.
        """
        if food_name not in self.data.index or volume_m3 <= 0:
            return 0.0

        # Convert density from g/cm³ to g/m³: multiply by 1,000,000
        density_g_per_m3 = self.data.loc[food_name, "Density (g/cm³)"] * 1_000_000
        calories_per_g = self.data.loc[food_name, "Calories Per Gram (kcal/g)"]

        mass_g = density_g_per_m3 * volume_m3
        total_calories = mass_g * calories_per_g

        return total_calories
    

if __name__ == "__main__":
    # Example usage
    estimator = CalorieEstimator(CALORIES_DATA_CSV_PATH)
    food_name = "juice-apple"
    volume_m3 = 0.001  # 1 liter
    calories = estimator.estimate(food_name, volume_m3)
    print(f"Estimated calories for {volume_m3} m³ of {food_name}: {calories:.2f} kcal")
