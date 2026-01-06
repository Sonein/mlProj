import xml.etree.ElementTree as ET
import re
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC

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

#per sample normalization, this was used everywhere previously but caused issues, here it somehow produces better results
def normalize_features(features):
    offset = 0
    normalized = []

    for descriptor, size in DESCRIPTOR_SIZES.items():
        block = features[offset:offset + size]

        if descriptor in {"ScalableColorType", "ColorStructureType", "EdgeHistogramType"}:
            block = normalize(block.reshape(1, -1), norm="l1")[0]
        else:
            block = StandardScaler().fit_transform(block.reshape(-1, 1)).flatten()

        normalized.append(block)
        offset += size

    return np.concatenate(normalized)

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

#per-descriptor pca
def fit_descriptor_pca(X_train_local, n_components_ratio=0.85):
    pca_models = {}
    scalers = {}
    reduced_dims = {}

    for named, sl in descriptor_slices.items():
        block = X_train_local[:, sl]

        #scaling per descriptor
        scaler = StandardScaler()
        block_scaled = scaler.fit_transform(block)

        #define pca with variance retention
        pca = PCA(
            n_components=n_components_ratio,
            whiten=True,
            random_state=42
        )
        block_pca = pca.fit_transform(block_scaled)

        pca_models[named] = pca
        scalers[named] = scaler
        reduced_dims[named] = block_pca.shape[1]

        print(f"{named}: {block.shape[1]} → {block_pca.shape[1]}")

    return pca_models, scalers, reduced_dims

def transform_with_pca(X, pca_models, scalers):
    transformed_blocks = []

    for name, sl in descriptor_slices.items():
        block = X[:, sl]
        block_scaled = scalers[name].transform(block)
        block_pca = pca_models[name].transform(block_scaled)
        transformed_blocks.append(block_pca)

    return np.concatenate(transformed_blocks, axis=1)

splits = [403, 501, 410, 594, 482, 137]
idp = Path("path to dataset-resized")
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
    numbers = normalize_features(numbers)
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

#descriptor slicing for per-descriptor pca
descriptor_slices = {}
offset = 0
for name, size in DESCRIPTOR_SIZES.items():
    descriptor_slices[name] = slice(offset, offset + size)
    offset += size

pca_models, scalers, reduced_dims = fit_descriptor_pca(X_train)

X_train_pca = transform_with_pca(X_train, pca_models, scalers)
X_test_pca  = transform_with_pca(X_test,  pca_models, scalers)

#log. regression definition
clf = LogisticRegression(
    max_iter=5000,
    C=1.0,
    solver="lbfgs",
    multi_class="auto",
    n_jobs=-1
)
#linearscv definition
clf2 = LinearSVC(
    C=1.0,
    max_iter=5000
)

#training and testing of both
clf.fit(X_train_pca, y_train)
clf2.fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca)

print("Accuracy:", accuracy_score(y_test, y_pred))
#print(classification_report(y_test, y_pred)) #57.3

y_pred2 = clf2.predict(X_test_pca)

print("Accuracy:", accuracy_score(y_test, y_pred2))
#print(classification_report(y_test, y_pred2)) #61
