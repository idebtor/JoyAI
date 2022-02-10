"""
@author: Robert Kamunde
"""

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from tensorflow.keras.models import load_model

#creating a list of all the foods, in the argument i put the path to the folder that has all folders for food
def create_foodlist(path):
    list_ = list()
    for root, dirs, files in os.walk(path, topdown=False):
      for name in dirs:
        list_.append(name)
    return list_    

#loading the model i trained and finetuned        
my_model = load_model('model_trained.h5', compile = False)
food_list = create_foodlist("food-101/images")



allowance = 2000
ingest = []


#function to help in predicting classes of new images loaded from my computer(for now) 
def predict_class(model, images, show = True):
    caloric_intake = 0
    for img in images:
        img = image.load_img(img, target_size=(299, 299))
        img = image.img_to_array(img)                    
        img = np.expand_dims(img, axis=0)         
        img /= 255.                                      

        pred = model.predict(img)
        index = np.argmax(pred)    #Returns the indices of the maximum values along an axis, In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
        food_list.sort()
        pred_value = food_list[index]
        if show:
            plt.imshow(img[0])   
            plt.axis('off')
            plt.title(pred_value)
            plt.show()
        kcal = nutrition_information(pred_value)
        ingest.append(pred_value)
        caloric_intake += int(kcal)
        calculate_calories(ingest, caloric_intake)

def nutrition_information(pred_value):
    with open("nutrition.txt") as f:
        for food in f:
            food_info = food.split(',')
            foodname = food_info[0]
            calories = food_info[1]
            carbs = food_info[2]
            protein = food_info[3]
            fats = food_info[4]
            recommendedfood = food_info[5].strip()
            if pred_value == foodname:
                print("Calories =",calories)
                labels = 'carbs', 'protein', 'fats'
                sizes = [carbs, protein, fats]
                colors = ['#ff9999','#ffc000','#8fd9b6']
                wedgeprops= {'width':0.7, 'edgecolor':'w', 'linewidth':5}
                fig1, ax1 = plt.subplots()
                explode = (0.05,0.05,0.05)
                ax1.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%',shadow=True, startangle=90, colors=colors, wedgeprops=wedgeprops)
                ax1.axis('equal')
                plt.title("Nutrition Balance")
                plt.axis('off')
                plt.show()
                fig = plt.imshow(mpimg.imread(recommendedfood+'.jpg'))
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                print("Recommended side dish:",recommendedfood)
                plt.axis('off')
                plt.show()
                return calories
        

def calculate_calories(pred_value, caloric_intake):
    #allowance = 2000
    #ingest = []
    #caloric_intake = 0
    #ingest.append(pred_value)
    #caloric_intake += int(calories)
    print('Until now, you have ingested', ingest, '.\nTotal caloric intake:', caloric_intake)
    if caloric_intake < allowance:
        print('You can have', allowance-caloric_intake, 'kcal more for today.')
    elif caloric_intake == allowance:
        print('You have ingested just enough amount for today.')
    else:
        print('You had too much! You exceeded', caloric_intake-allowance, 'kcal.')
    

            
images = []
images.append('sample1.jpeg')
#images.append('sample2.jpeg')
images.append('sample3.jpeg')
images.append('sample4.jpeg')
images.append('sample5.jpeg')
images.append('sample6.jpeg')
images.append('sample7.jpeg')
images.append('sample8.jpeg')
images.append('sample9.jpeg')
images.append('sample10.jpeg')


print("PREDICTIONS BASED ON PICTURES UPLOADED")
predict_class(my_model, images, True)