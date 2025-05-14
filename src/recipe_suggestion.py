import pandas as pd
import cv2
from segmentation_model import SegmentationModel

RECIPE_XLSX_PATH = "../data/food_recipes.xlsx"  # Update this path to your actual Excel file


class FoodRecipeSuggester(SegmentationModel):
    def __init__(self):
        super().__init__()
        # Load Excel file and set index to "Class Name"
        self.recipe_data = pd.read_excel(RECIPE_XLSX_PATH)
        self.recipe_data.set_index("Class Name", inplace=True)

    def get_recipe(self, food_item: str) -> str:
        """
        Get the recipe (ingredients and steps) for a given food item.
        Args:
            food_item (str): Name of the detected food item.
        Returns:
            str: Recipe details if found, else a message.
        """
        if food_item in self.recipe_data.index:
            ingredients = self.recipe_data.loc[food_item, "Ingredients"]
            steps = self.recipe_data.loc[food_item, "Steps"]
            return f"Ingredients: {ingredients}\nSteps: {steps}"
        else:
            return "No recipe found for this item."

    def suggest_recipes_from_image(self, image_path: str):
        """
        Run segmentation on the image and print recipes for each detected food item.
        Args:
            image_path (str): Path to the image file.
        """
        image = cv2.imread(image_path)
        results = self.segment(image)
        object_names = self.get_object_names(results)

        if not object_names:
            print("No recognizable food items found.")
            return

        print("Detected food items and recipes:\n")
        for food in object_names:
            recipe = self.get_recipe(food)
            print(f"{food} ➜\n{recipe}\n")

if __name__ == "__main__":
    suggester = FoodRecipeSuggester()
    image_path = "../images/147426.jpg"
    suggester.suggest_recipes_from_image(image_path)
