import xml.etree.ElementTree as ET
import numpy as np
import re
import torch.nn as nn
import torchvision.models as models
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

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

#evaluation function for the current iteration of the model to get current testing accuracy
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, mpeg7, labels in dataloader:
            images = images.to(device)
            mpeg7 = mpeg7.to(device)
            labels = labels.to(device)

            outputs = model(images, mpeg7)
            _, preds = torch.max(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

#training function for the current epoch
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, mpeg7, labels in loader:
        images = images.to(device)
        mpeg7 = mpeg7.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images, mpeg7)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

#definition of the mpeg7 mlp branch with a previous model definition as well
#if you wanna use the previous model you will need to adjust the input size of the fusion classifier (+128 not +64)
class MPEG7Branch(nn.Module):
    '''def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(737, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU()
        )''' #baseline best 72

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(737, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU()
        ) #this got us 71 to 72 percent, no change

    def forward(self, x):
        return self.net(x)

#the fusion classifier with the pretrained resnet
class DescriptorResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        # Image branch
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # output = 2048

        # MPEG-7 branch
        self.mpeg7 = MPEG7Branch()

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 64, 512), #was 128 before 64 for idea A
            nn.BatchNorm1d(512), #added by as an adjustment, can be removed
            nn.ReLU(),
            nn.Dropout(0.6), #original was 0.5 adjusted to 0.6
            nn.Linear(512, num_classes) #stable 72 percent
        )

    def forward(self, image, mpeg7):
        img_feat = self.resnet(image)
        mpeg7_feat = self.mpeg7(mpeg7)
        fused = torch.cat([img_feat, mpeg7_feat], dim=1)
        return self.classifier(fused)

#definition of a mixed dataset comprised of images for resnet nad descriptors for the mlp
class WasteDataset(Dataset):
    def __init__(self, image_paths, mpeg7_features, labels, transform=None):
        self.image_paths = image_paths
        self.X = mpeg7_features
        self.y = labels
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        mpeg7 = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)

        return img, mpeg7, label

splits = [403, 501, 410, 594, 482, 137]
idp = Path("path to dataset-resized")
IMAGE_REGEX = re.compile(r".*\.(xml)$", re.IGNORECASE)
xmls = [f for f in idp.rglob("*") if IMAGE_REGEX.match(f.name) and (len(f.relative_to(idp).parts) <= 3)]
IMAGE_REGEX = re.compile(r".*\.(jpg)$", re.IGNORECASE)
images = [f for f in idp.rglob("*") if IMAGE_REGEX.match(f.name) and (len(f.relative_to(idp).parts) <= 3)]
image_count = len(xmls)

dataset = []
masks = []
categories_d = []
i = 0
c = 0

#prepares the dataset
for image_file in xmls:
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
images = np.array(images)

rng = np.random.default_rng(seed=439)
indices = rng.permutation(len(X))

X = X[indices]
y = y[indices]
Ximg = images[indices]
masks = masks[indices]
#transformer definitions
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#test and train split
split_idx = int(0.65 * len(X))

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
Ximg_train, Ximg_test = Ximg[:split_idx], Ximg[split_idx:]

X_train = X_train.astype(np.float32)
X_test  = X_test.astype(np.float32)

X_train = np.clip(X_train, -3, 3)
X_test  = np.clip(X_test, -3, 3)

#scaling the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

#final complex dataset definition
train_dataset = WasteDataset(
    image_paths=Ximg_train,
    mpeg7_features=X_train,
    labels=y_train,
    transform=train_transform
)

test_dataset = WasteDataset(
    image_paths=Ximg_test,
    mpeg7_features=X_test,
    labels=y_test,
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#instancing the model
model = DescriptorResNet(num_classes=6).to(device)

#setting up optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

#freeze ResNet initially
for p in model.resnet.parameters():
    p.requires_grad = False

num_epochs = 17

#training loop
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )
    #unfreeze resnet after few epochs
    if epoch == 10:
        for p in model.resnet.layer4.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-5
        )

    test_acc = evaluate(model, test_loader, device)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Test Acc: {test_acc:.4f}"
    ) #between 71 and 73