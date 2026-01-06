import xml.etree.ElementTree as ET
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler, normalize
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

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

#mlp definition
class ShallowMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.fc_out(x)


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

X_train = X_train.astype(np.float32)
X_test  = X_test.astype(np.float32)

X_train = np.clip(X_train, -3, 3)
X_test  = np.clip(X_test, -3, 3)

#scaling the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

#conversion for torch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256)

device = "cuda" if torch.cuda.is_available() else "cpu"

#instancing the model
model = ShallowMLP(input_dim=X_train.shape[1], num_classes=len(set(y_train)))
model.to(device)

#setting up optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0
best_model = None
patience = 10
patience_counter = 0

#training loop
for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    acc = correct / total
    print(f"Epoch {epoch:03d} | Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_model = deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break

model.load_state_dict(best_model)
print("Best accuracy:", best_acc)

#67.3 percent