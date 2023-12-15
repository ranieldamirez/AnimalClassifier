# This script serves to translate the dataset derived from Kaggle
# dataset = https://www.kaggle.com/datasets/alessiocorrado99/animals10


import os

translate = {"cane": "dog", "ragno": "spider", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel"}
directory_path = "./raw-img"

def rename_files(directory, legend):
    # Check if directory is valid
    if not os.path.isdir(directory):
        print("Directory not found:", directory)
        return
    
    else:
        # Iterate over all files in the directory
        for name in os.listdir(directory):
            old_name =  os.path.join(directory, name)

            if os.path.isdir(old_name):
                try:
                    new_name = translate[name]
                    new_name = os.path.join(directory, new_name)
                    os.rename(old_name, new_name)
                    print("{0} image training source identified".format(new_name))
                except KeyError:
                    continue
rename_files(directory_path, translate)
