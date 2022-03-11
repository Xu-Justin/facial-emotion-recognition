import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

emotions = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

class Dataset():
    
    image_shape = (224, 224)
    
    base_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(image_shape),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    aug_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.RandomEqualize(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, translate = (0.2, 0.2)),
        transforms.RandomRotation(20),
        
        transforms.Resize(image_shape),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def loader(self, path, batch_size, augmentation=False):
        transform = self.aug_transform if augmentation else self.base_transform
        data = datasets.ImageFolder(path, transform=transform)
        assert(data.class_to_idx == emotions)
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        return loader
        
    def init_trainloader(self, path):
        self.trainloader = self.loader(path, self.batch_size, augmentation=True)
    
    def init_validationloader(self, path):
        self.validationloader = self.loader(path, self.batch_size)
    
    def init_testloader(self, path):
        self.testloader = self.loader(path, self.batch_size)

def init_model(pretrained=True):
    model = models.vgg16(pretrained=pretrained)
    model.classifier[6] = nn.Linear(4096, 7)
    return model

def load_weight(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Loaded model weight from {path}")

def save_weight(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved model weight to {path}")

def eval(model, loader, device):
    model.eval()

    y_pred = torch.empty(0).to(device)
    y_true = torch.empty(0).to(device)

    with torch.no_grad():
        for X, y in tqdm(loader, desc=f'Eval'):
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            pred = logits.argmax(1)

            y_pred = torch.cat((y_pred, pred))
            y_true = torch.cat((y_true, y))
    
    return y_pred, y_true

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--BATCH_SIZE', type=int, required=True,help='')
parser.add_argument('--PATH_DATASET', type=str, required=True, help='')
parser.add_argument('--PATH_MODEL_WEIGHT', type=str, required=True, help='')
parser.add_argument('--CPU_ONLY', action='store_true', help='')
args = parser.parse_args()

def main():

    PATH_DATASET = args.PATH_DATASET
    PATH_MODEL_WEIGHT = args.MODEL_WEIGHT
    BATCH_SIZE = args.BATCH_SIZE
    CPU_ONLY = args.CPU_ONLY

    print()
    print(f"Path dataset      : {PATH_DATASET}")
    print(f"Path model weight : {PATH_MODEL_WEIGHT}")
    print(f"Batch size        : {BATCH_SIZE}")
    print(f"CPU only    : {CPU_ONLY}")
    print()

    # DATASET
    data = Dataset(BATCH_SIZE)
    data.init_testloader(PATH_DATASET)

    print("Number of sample :",len(data.testloader.sampler))
    
    # DEVICE
    if(CPU_ONLY):
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    if(device.type == 'cuda'):
       torch.cuda.empty_cache()

    # MODEL
    if(PATH_MODEL_WEIGHT is not None):
        model = init_model(pretrained=False)
        load_weight(model, PATH_MODEL_WEIGHT)
    else:
        model = init_model()
    model.to(device)

    print()

    # EVAL
    y_pred, y_true = eval(model, data.testloader, device)
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()

    print()

    # Confusion Matrix
    cf = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix")
    print(cf)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("Accuracy  : %.2f%%"%(acc*100))
    print("Precision : %.4f"%(precision))
    print("Recall    : %.4f"%(recall))
    print("F1-Score  : %.4f"%(f1))

if __name__ == '__main__':
    main()