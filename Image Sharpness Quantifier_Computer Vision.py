# Importing Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# File Directory
def load_dataset(directory):
    for filename in os.listdir(directory):
        image_path=os.path.join(directory,filename)

# Metode Laplacian untuk menghitung matrix ketajaman suatu gambar dari dataset
def compute_sharpness(directory):
    
    # List Dataframe
    sharpness_val=[]
    filename_list=[]
    
    try:
        # Membaca gambar dari path
        for filename in os.listdir(directory):
            image_path=os.path.join(directory,filename)
            image = cv2.imread(image_path)
        
            if image is not None:
                GRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(GRAY, cv2.CV_64F)
                sharpness = laplacian.var()
                
                sharpness_val.append(sharpness)
                filename_list.append(filename)
                        
# Jika Gambar tidak dapat dibaca
    except Exception as e:
        print(f"Error: {str(e)}")
        
    return sharpness_val, filename_list
    
# Metode Gradien Sobel untuk menghitung matrix ketajaman suatu gambar dari dataset
def compute_sharpness_sobel(directory):
    
    # List Dataframe
    sharpness_val=[]
    filename_list=[]
    
    try:
        # Membaca gambar dari path
        for filename in os.listdir(directory):
            image_path=os.path.join(directory,filename)
            image = cv2.imread(image_path)
            image = cv2.imread(image_path)
        
            if image is not None:
                GREY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
         # Menghitung gradien gambar menggunakan metode Sobel
                grad_x = cv2.Sobel(GREY, cv2.CV_64F, 1, 0, ksize=5)
                grad_y = cv2.Sobel(GREY, cv2.CV_64F, 0, 1, ksize=5)
            
        # Menghitung magnitude gradien
                magnitude = cv2.magnitude(grad_x, grad_y)
                sharpness = magnitude.mean()
            
                sharpness_val.append(sharpness)
                filename_list.append(filename)
             
# Gambar tidak dapat dibaca
    except Exception as e:
        print(f"Error: {str(e)}")
        
    return sharpness_val, filename_list
 
# Main Function
directory1 = './Dataset UTS Comvis/PCB TYPE1/'
directory2 = './Dataset UTS Comvis/PCB TYPE2/'

sharpness1, file_name1 = compute_sharpness(directory1)
sharpness2, file_name2 = compute_sharpness(directory2)

sobel1, file_name1 = compute_sharpness(directory1)
sobel2, file_name2 = compute_sharpness(directory2)

# DataFrame Column
df1 = pd.DataFrame(
    {
        "Type1_FileName" : file_name1,
        "Sharpness" : sharpness1,
        "Sobel" : sobel1,       
    }
).sort_values(by="Sharpness", ascending=False, ignore_index=True)


df2 = pd.DataFrame(
    {
        "Type2_FileName" : file_name2,
        "Sharpness" : sharpness2,
        "Sobel" : sobel2,       
    }
).sort_values(by="Sharpness", ascending=False, ignore_index=True)

highest_sharpness1 = df1.iloc[0]["Sharpness"] 
highest_sharpness2 = df2.iloc[0]["Sharpness"]

# Print Image
im1 = cv2.imread('./Dataset UTS ComVis/PCB TYPE1/WIN_20230125_20_22_02_Pro.jpg')
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(im1)

im2 = cv2.imread('./Dataset UTS ComVis/PCB TYPE2/WIN_20230125_21_04_06_Pro.jpg')
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(im2)

print(df1)
print("\n")
print(df2)
print("\n")
plt.show()