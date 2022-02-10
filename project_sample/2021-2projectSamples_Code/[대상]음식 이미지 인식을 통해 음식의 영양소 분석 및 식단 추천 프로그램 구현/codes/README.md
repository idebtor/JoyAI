# Food101Project
* **The dataset being used is [Food 101] (download it here https://www.kaggle.com/dansbecker/food-101)** 
- **This dataset has 101000 images in total. It's a food dataset with 101 categories(multiclass)**
  - **Each type of food has 750 training samples and 250 test samples** 
  - **The entire dataset is about 5GB in size** 
* **Use tensorflow-gpu for faster training time** 

* **You need to have the food-101 dataset in your working directory (otherwise you should change the paths to the food-101 file)** 
- Run food101Work.py to create the train set and test set from the dataset.
- Run Food101Model.py to tune and train the model ( a MobileNetV2 Pretrained model is used and tuned to the food-101 dataset ) 
- Run Prediction_food_images.py to predict new food images. ( the images to be predicted here were in the working directory, if you have them in another location you should change the path)
