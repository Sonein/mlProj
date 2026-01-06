import xml.etree.ElementTree as ET
import re
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

MPEG7_NS = {
    'mpeg7': 'http://www.mpeg7.org/2001/MPEG-7_Schema',
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
}

DESCRIPTOR_SIZES = {
    "ScalableColorType": 256,
    "DominantColorType": 33,
    "ColorLayoutType": 12,
    "ColorStructureType": 256,
    "HomogeneousTextureType": 62,
    "EdgeHistogramType": 80,
    "ContourShapeType": 3,
    "RegionShapeType": 35,
}

#function for xml parsing of each descriptor
def _safe_float_list(text):
    if text is None:
        return []
    return [float(x) for x in text.strip().split() if x]

def _pad_or_truncate(values, size):
    values = values[:size]
    if len(values) < size:
        values = values + [0.0] * (size - len(values))
    return np.array(values, dtype=np.float32)


def extract_descriptor(root, descriptor_name, max_size):
    xpath = f".//mpeg7:Descriptor"
    elements = root.findall(xpath, MPEG7_NS)

    if not elements:
        return np.zeros(max_size, dtype=np.float32), 0

    values = []

    for elem in elements:
        descriptor_type = elem.attrib.get('{http://www.w3.org/2000/10/XMLSchema-instance}type')
        if descriptor_type != descriptor_name:
            continue

        if descriptor_type == 'ScalableColorType':
            coeff_text = elem.find('mpeg7:Coeff', MPEG7_NS)
            if coeff_text.text:
                values.extend(_safe_float_list(coeff_text.text))
            break

        elif descriptor_type == 'DominantColorType':
            for val in elem.findall('.//mpeg7:Value', MPEG7_NS):
                for child in val:
                    if child.text:
                        values.extend(_safe_float_list(child.text))
            break

        elif descriptor_type == 'ColorLayoutType':
            for tag in ['YDCCoeff', 'CbDCCoeff', 'CrDCCoeff', 'YACCoeff5', 'CbACCoeff2', 'CrACCoeff2']:
                coeff = elem.find(f'mpeg7:{tag}', MPEG7_NS)
                if coeff is not None and coeff.text:
                    values.extend(_safe_float_list(coeff.text))
            break

        elif descriptor_type == 'ColorStructureType':
            vals = elem.find('mpeg7:Values', MPEG7_NS)
            if vals is not None and vals.text:
                values.extend(_safe_float_list(vals.text))
            break

        elif descriptor_type == 'HomogeneousTextureType':
            for tag in ['Average', 'StandardDeviation', 'Energy', 'EnergyDeviation']:
                val = elem.find(f'mpeg7:{tag}', MPEG7_NS)
                if val is not None and val.text:
                    values.extend(_safe_float_list(val.text))
            break

        elif descriptor_type == 'EdgeHistogramType':
            bins = elem.find('mpeg7:BinCounts', MPEG7_NS)
            if bins is not None and bins.text:
                values.extend(_safe_float_list(bins.text))
            break

        elif descriptor_type == 'ContourShapeType':
            for tag in ['GlobalCurvature', 'HighestPeakY']:
                val = elem.find(f'mpeg7:{tag}', MPEG7_NS)
                if val is not None and val.text:
                    values.extend(_safe_float_list(val.text))
            break

        elif descriptor_type == 'RegionShapeType':
            art = elem.find('mpeg7:MagnitudeOfART', MPEG7_NS)
            if art is not None and art.text:
                values.extend(_safe_float_list(art.text))
            break

    values = _pad_or_truncate(values, max_size)
    return values

#reads all relevant xml files in directory
def extract_mpeg7_features(xml_path):
    with open(xml_path, 'r') as f:
        xml_string = f.read()
    tree = ET.ElementTree(ET.fromstring(xml_string))
    root = tree.getroot()

    features = []
    masks_temp = []

    d=0

    for descriptor, size in DESCRIPTOR_SIZES.items():
        vec = extract_descriptor(root, descriptor, size)
        features.append(vec)
        masks_temp.extend([d] * len(vec))
        d += 1

    return np.concatenate(features), np.array(masks_temp, dtype=np.uint8)

splits = [403, 501, 410, 594, 482, 137]
idp = Path("/home/sonya/Documents/mlProj/dataset-resized/")
IMAGE_REGEX = re.compile(r".*\.(xml)$", re.IGNORECASE)
image_files = [f for f in idp.rglob("*") if IMAGE_REGEX.match(f.name) and (len(f.relative_to(idp).parts) <= 3)]
image_count = len(image_files)

dataset = []
masks = []
categories_d = []
i = 0
c = 0

#prepares the dataset
for image_file in image_files:
    numbers, mask = extract_mpeg7_features(image_file)
    dataset.append(numbers)
    masks.append(mask)
    categories_d.append(c)

    i += 1
    if i >= splits[c]:
        i = 0
        c += 1

X = np.array(dataset, dtype=np.float32)
y = np.array(categories_d, dtype=np.int64)
masks = np.array(masks, dtype=np.uint8)

rng = np.random.default_rng(seed=439)
indices = rng.permutation(len(X))

X = X[indices]
y = y[indices]
masks = masks[indices]

#test and train split
split_idx = int(0.65 * len(X))

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

#scaling the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

#Random Forest definition and training
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
) #67.2 percenta

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
#print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
#print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#LightGBM segment
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

#LightGBM definition and training
lgbm = lgb.LGBMClassifier(
    objective="multiclass",
    n_estimators=600,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42
) #best 68.6

lgbm.fit(X_train, y_train)

y_pred = lgbm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
#print("\nClassification Report:\n", classification_report(y_test, y_pred))
#print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#feature pruning according to importance
importances = lgbm.feature_importances_

top_idx = np.argsort(importances)[-20:][::-1]

for i in top_idx:
    print(f"Feature {i:03d}: importance={importances[i]}")

offset = 0
for name, size in DESCRIPTOR_SIZES.items():
    block_imp = importances[offset:offset + size].sum()
    print(f"{name}: {block_imp}")
    offset += size

N = 500
top_features = np.argsort(importances)[-N:]

X_train_pruned = X_train[:, top_features]
X_test_pruned = X_test[:, top_features]

#new lightlgbm based on the pruned dataset
lgbm_pruned = lgb.LGBMClassifier(
    objective="multiclass",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=48,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42
)

lgbm_pruned.fit(X_train_pruned, y_train)

y_pred = lgbm_pruned.predict(X_test_pruned)

print("Accuracy:", accuracy_score(y_test, y_pred))

#69.03