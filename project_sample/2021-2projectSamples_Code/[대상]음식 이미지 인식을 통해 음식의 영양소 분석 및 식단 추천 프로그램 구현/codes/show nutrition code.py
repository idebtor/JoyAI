def show_nutrition(pred_value):
    file = open("nutrition_"+pred_value+".txt", "r")
    nutrition = file.readlines()
    kcal = nutrition[0].strip()
    carbs = nutrition[1].strip()
    protein = nutrition[2].strip()
    fat = nutrition[3].strip()
    recommend = nutrition[4].strip()

    print(kcal, "Kcal")
    ratio = [carbs, protein, fat]
    labels = ["Carbohydrates", "Protein", "Fat"]
    colors = ['#ff9999', '#ffc000', '#8fd9b6']
    wedgeprops = {'width':0.7, 'edgecolor':'w', 'linewidth':5}

    plt.pie(ratio, labels=labels, autopct='%.1f%%', colors=colors, wedgeprops=wedgeprops)
    plt.title("Nutrition Ratio")
    #startangle=211, counterclock=False, 
    plt.show()
    
    print("We recommend you to have '"+recommend+"' with", pred_value)
    image = plt.imread('recommend_'+pred_value+'.jpeg')
    plt.axis("off")
    plt.imshow(image)
    
    file.close()
