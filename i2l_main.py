import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

images_folder = './data/datasets/I2L_140k/formula_images/'
file_info = './data/datasets/I2L_140k/training_56/data_props.pkl'

file_train = './data/datasets/I2L_140k/training_56/df_train.pkl'
file_valid = './data/datasets/I2L_140k/training_56/df_valid.pkl'
file_test = './data/datasets/I2L_140k/training_56/df_test.pkl'

info = pd.read_pickle(file_info)
print(info['id2word'])

df_train = pd.read_pickle(file_train)


print(df_train.iloc[0])

for row in df_train.iloc:
    img = plt.imread(images_folder + row['image'])
    print(row['image'], img.shape, row["latex_ascii"])
    plt.imshow(img)
    plt.show()

