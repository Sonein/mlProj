import re
import os
import time
import math
from pathlib import Path
from collections import Counter
import numpy as np
import subprocess

#parses 4 values from temporary files created by descriptor matching
def parse_floats_from_file(file_path):
    try:
        with open(file_path, "r") as file:
            floats = [float(num.strip(".#IND")) for num in file.read().strip().split()]
            ll = len(floats)
            if ll > 4 or ll == 3 or ll == 2 or ll < 1:
                raise ValueError(f"Expected 1 or 4 floats, but found {len(floats)} in {file_path}")
            return floats
    except FileNotFoundError:
        print(f"Error: {file_path} not found!")
        return None
    except ValueError as e:
        print(f"Error processing {file_path}: {e}")
        return None

#computes distance between two mpeg7 xml files
def d4d(f1, f2, output_dir, vdm_path, type_):
    cmd = [vdm_path, "-d", "DC", "-n1", f2, "-n2", f2, "-o", str(odp) + "\\temp.txt"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    temp_floats = parse_floats_from_file(str(output_dir) + "\\temp.txt")
    dc_lim = temp_floats[0]
    os.makedirs(output_dir, exist_ok=True)

    temp_1_path = str(odp) + "\\temp-1.txt"
    temp_2_path = str(odp) + "\\temp-2.txt"
    cmd = [vdm_path, "-d", "SC DC CL CST", "-n1", str(f1), "-n2", str(f2), "-o", temp_1_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cmd = [vdm_path, "-d", "HT EH CS RS", "-n1", str(f1), "-n2", str(f2), "-o", temp_2_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    temp_1_floats = parse_floats_from_file(temp_1_path)
    temp_2_floats = parse_floats_from_file(temp_2_path)
    float_array = temp_1_floats + temp_2_floats
    float_array[1] = abs(float_array[1]-dc_lim)

    suma = 0
    if (type_ == "euc"):
        for floatt in float_array:
            suma += floatt*floatt
        suma = math.sqrt(suma)
    elif (type_ == "max"):
        suma = max(float_array)
    else:
        for floatt in float_array:
            suma += floatt
    
    return suma


splits = [403, 501, 410, 594, 482, 137]

idp = Path("absolute path to dataset-resized")
odp = Path("absolute path to output directory")
vdm = Path("absolute path to vdm.exe")
os.makedirs(odp, exist_ok=True)
IMAGE_REGEX = re.compile(r".*\.(xml)$", re.IGNORECASE)
image_files = [f for f in idp.rglob("*") if IMAGE_REGEX.match(f.name) and (len(f.relative_to(idp).parts) <= 3)]
image_count = len(image_files)

dataset = []
categories_d = []

#read all xml files in directory and append its path into the dataset, and give it a correct category
#category giving assumes the order of files is the same as in the dataset
if image_count == 0:
        print(f"No matching images found in {idp}, skipping...")
else:
    print(f"Adding {image_count} images to the matrix and making dataset...")
    i = 0
    c = 0
    for image_file in image_files:
        if i < splits[c] :
            dataset.append(image_file)
            categories_d.append(c)

        i += 1
        if i >= splits[c]:
             i = 0
             c += 1

X = np.array(dataset)
y = np.array(categories_d)
rng = np.random.default_rng(seed=42)
indices = rng.permutation(len(X))
X = X[indices]
y = y[indices]
#do the 65 35 split
split_idx = int(0.65 * len(X))

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

test_all = 0
test_corr = 0

#classify the testing set on the training
#returns the accuracy it achieved so far
for i in range(len(X_test)):
    start_time = time.time()
    top = [[0, 1000000], [0, 1000000], [0, 1000000], [0, 1000000], [0, 1000000], [0, 1000000], [0, 1000000], [0, 1000000], [0, 1000000], [0, 1000000]]
    for j in range(len(X_train)):
        s = d4d(X_test[i], X_train[j], odp, vdm, "euc")
        cat = y_train[j]
        for eeach in top:
            if eeach[1] >= s:
                temp1 = eeach[0]
                temp = eeach[1]
                eeach[0] = cat
                eeach[1] = s
                cat = temp1
                s = temp
    first_values = [x[0] for x in top]
    most_common_value, count = Counter(first_values).most_common(1)[0]
    test_all += 1
    if(most_common_value == y_test[i]):
        test_corr += 1        
    elapsed_time = round(time.time() - start_time, 2)
    print(f"Time:({elapsed_time}), current accuracy:({test_corr/test_all}), left to do:({len(X_test)-i-1})")