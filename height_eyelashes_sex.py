
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sex = pd.read_csv("height_eyelashes_sex.csv", comment="#")
sex['color'] = np.where(sex.sex == "male", "blue", "pink")

plt.scatter(sex.eyelash_length_mm, sex.height_inches, color=sex.color)
#y=2.278x+48.246

def predictor(eyelash_length_mm, height_inches):
    y = 2.3 * eyelash_length_mm + 48
    if y < height_inches:
        return "male"
    else:
        return "female"

print(predictor(10, 5*12 + 0))
print(predictor(1, 6*12 + 2))
