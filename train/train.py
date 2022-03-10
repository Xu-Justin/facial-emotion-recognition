import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from tqdm import tqdm

emotions = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

class Dataset():
    
    image_shape = (224, 224)
    
    base_transform = transforms.Compose([
        transforms.Resize(image_shape),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    aug_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.RandomEqualize(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(image_shape),
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

def init_model():
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 7)
    return model

def load_weight(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Loaded model weight from {path}")

def save_weight(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved model weight to {path}")

def train(model, epochs, trainloader, evalloader, criterion, optimizer, scheduler, device):
    
    for epoch in range(epochs):
        
        model.train()
        
        running_loss = 0
        running_acc = 0

        for X, y in tqdm(trainloader, desc=f'Train Epoch - {epoch+1}'):
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(X) / len(trainloader.sampler)
            running_acc += ( (logits.argmax(1)==y).sum() / len(y) ) * len(X) / len(trainloader.sampler)

        model.eval()

        eval_loss = 0
        eval_acc = 0

        with torch.no_grad():
            for X, y in tqdm(evalloader, desc=f'Eval Epoch - {epoch+1}'):
                X = X.to(device)
                y = y.to(device)

                logits = model(X)
                loss = criterion(logits, y)

                eval_loss += loss.item() * len(X) / len(evalloader.sampler)
                eval_acc += ( (logits.argmax(1)==y).sum() / len(y) ) * len(X) / len(evalloader.sampler)
        
        scheduler.step()

        print()
        print(f"Epoch - {epoch+1}")
        print(f"Training Loss   : {running_loss:.6f}")
        print(f"Training Acc    : {running_acc:.6f}")
        print(f"Validation Loss : {eval_loss:.6f}")
        print(f"Validation Acc  : {eval_acc:.6f}")
        print()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--EPOCHS', type=int, required=True,help='')
parser.add_argument('--BATCH_SIZE', type=int, required=True,help='')
parser.add_argument('--PATH_TRAIN_DATASET', type=str, required=True, help='')
parser.add_argument('--PATH_EVAL_DATASET', type=str, required=True, help='')
parser.add_argument('--PATH_MODEL_WEIGHT', type=str, help='', const=None)
parser.add_argument('--PATH_OUTPUT_MODEL_WEIGHT', type=str, required=True, help='')
parser.add_argument('--CPU_ONLY', action='store_true', help='')
parser.add_argument('--CLASSIFIER_ONLY', action='store_true', help='')
args = parser.parse_args()

def main():

    BATCH_SIZE = args.BATCH_SIZE
    PATH_TRAIN_DATASET = args.PATH_TRAIN_DATASET
    PATH_EVAL_DATASET = args.PATH_EVAL_DATASET

    CPU_ONLY = args.CPU_ONLY

    PATH_MODEL_WEIGHT = args.PATH_MODEL_WEIGHT

    CLASSIFIER_ONLY = args.CLASSIFIER_ONLY
    EPOCHS = args.EPOCHS

    PATH_OUTPUT_MODEL_WEIGHT = args.PATH_OUTPUT_MODEL_WEIGHT

    print()
    print(f"Epochs     : {EPOCHS}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Path train dataset       : {PATH_TRAIN_DATASET}")
    print(f"Path eval dataset        : {PATH_EVAL_DATASET}")
    print(f"Path model weight        : {PATH_MODEL_WEIGHT}")
    print(f"Path output model weight : {PATH_OUTPUT_MODEL_WEIGHT}")
    print(f"CPU only        : {CPU_ONLY}")
    print(f"Classifier only : {CLASSIFIER_ONLY}")
    print()

    # DATASET
    data = Dataset(BATCH_SIZE)
    data.init_trainloader(PATH_TRAIN_DATASET)
    data.init_validationloader(PATH_EVAL_DATASET)

    print("Number of train sample :",len(data.trainloader.sampler))
    print("Number of eval sample  :",len(data.validationloader.sampler))

    # DEVICE
    if(CPU_ONLY):
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    if(device.type == 'cuda'):
       torch.cuda.empty_cache()

    # MODEL
    model = init_model()
    if(PATH_MODEL_WEIGHT is not None):
        load_weight(model, PATH_MODEL_WEIGHT)
    model.to(device)

    print()

    try:
        # TRAIN
        criterion = nn.CrossEntropyLoss()
        if(CLASSIFIER_ONLY):
            for param in model.features.parameters():
                param.requires_grad = False

            optimizer = torch.optim.SGD(model.classifier.parameters(), lr=1e-3, momentum=0.9, nesterov=True, weight_decay=0.0001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.75)

        else:
            for param in model.features.parameters():
                param.requires_grad = True

            optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, nesterov=True, weight_decay=0.0001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.75, verbose=True)

        train(model, EPOCHS, data.trainloader, data.validationloader, criterion, optimizer, scheduler, device)
 
    except Exception as e:
        print(e)

    finally:
        # SAVE MODEL
        save_weight(model, PATH_OUTPUT_MODEL_WEIGHT)

if __name__ == '__main__':
    main()