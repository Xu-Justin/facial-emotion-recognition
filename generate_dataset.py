import os, shutil
import zipfile
import kaggle
import pandas as pd

from tqdm import tqdm
from torchvision.utils import save_image
from torch import tensor

class FER2013:

    image_shape = (48, 48)
    
    labels = {
        0 : 'angry',
        1 : 'disgust',
        2 : 'fear',
        3 : 'happy',
        4 : 'sad',
        5 : 'surprise',
        6 : 'neutral'
    }

    emotions = dict()
    
    for label, emotion in labels.items():
        emotions[emotion] = label
    
    @classmethod
    def label2emotion(cls, label):
        return cls.labels[label]
    
    @classmethod
    def emotion2label(cls, emotion):
        return cls.emotions[emotion]

def extract(source, target):
    with zipfile.ZipFile(source, 'r') as zip_ref:
        zip_ref.extractall(target)

def generate(df, path):

    os.makedirs(path)
    for emotion in FER2013.emotions:
        os.makedirs(os.path.join(path, emotion))

    for index, label, pixels in tqdm(df.itertuples(), desc=path): 
        emotion = FER2013.label2emotion(label)
        pixels = tensor( [float(pixel)/255 for pixel in pixels.split()] ).view(FER2013.image_shape)
        save_image(pixels, os.path.join(path, emotion, f"{index}.jpg"))

def report(path):
    count_sum = 0
    count_emotions = dict()
    for emotion in FER2013.emotions:
        count = len(os.listdir( os.path.join(path, emotion) ))
        count_emotions[emotion] = count
        count_sum += count
    return count_sum, count_emotions

def main(path, clean=False):

    if(os.path.exists(path)):
        if(clean): shutil.rmtree(path)
        else: raise Exception(f"Path already exists. Received: {path}")

    os.makedirs(path)

    # Download dataset from Kaggle
    competition = 'challenges-in-representation-learning-facial-expression-recognition-challenge'
    file_name = 'icml_face_data.csv'

    kaggle.api.authenticate()
    kaggle.api.competition_download_file(competition, 'icml_face_data.csv', path)
    extract(os.path.join(path, f'{file_name}.zip'), path)
    os.remove(os.path.join(path, f'{file_name}.zip'))
    
    # Load csv
    df = pd.read_csv( os.path.join(path, file_name))
    df = df.rename(columns={' Usage':'Usage'})
    df = df.rename(columns={' pixels':'pixels'})

    train_df = df[df['Usage'] == 'Training'][['emotion', 'pixels']]
    val_df = df[df['Usage'] == 'PublicTest'][['emotion', 'pixels']]
    test_df = df[df['Usage'] == 'PrivateTest'][['emotion', 'pixels']]
    
    del df

    # Save dataframes to images
    generate(train_df, os.path.join(path, 'train/'))
    generate(val_df, os.path.join(path, 'val/'))
    generate(test_df, os.path.join(path, 'test/'))

    del train_df, val_df, test_df

    # Validate Dataset
    print('Training', report( os.path.join(path, 'train') ))
    print('PublicTest', report( os.path.join(path, 'val') ))
    print('PrivateTest', report( os.path.join(path, 'test') ))

if __name__ == '__main__':
    main(path='./dataset/fer2013/')
    print("Finished.")