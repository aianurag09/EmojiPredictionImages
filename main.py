from model import model
import cv2
import os
import numpy as np
import argparse
import json
import fasttext

from keras import backend as K
from keras.optimizers import SGD
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
fasttext_model = fasttext.load_model('model.bin')

with open('mappings.json', 'r') as f:
    coco_descriptions = json.load(f)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def emoji_names():
    with open('emojis.json', 'r') as f:
        parsed = json.load(f)
    
    senses = []
    for x in range(len(parsed)):
        vector = np.zeros((100, ))
        tokenized = tokenizer.tokenize(parsed[x]['name'])
        for word in tokenized:
            vector = vector + np.array(fasttext_model[word])
        vector = vector / len(tokenized)
        senses.append(vector)
    return senses

def sense_definitions():
    with open('emojis.json', 'r') as f:
        parsed = json.load(f)

    SenseLabels = []
    senses = ['nouns','adjectives','verbs']
    for x in range(len(parsed)):
        tempExamples = []
        for sense in senses:
            for i in range(0,len(parsed[x]['senses'][sense])):
                for key in parsed[x]['senses'][sense][i]:
                    for sentence in parsed[x]['senses'][sense][i][key]['definitions']:
                        tempExamples.append(sentence)
        BagOfWordsExamples = {}
        for example in tempExamples:
            tokenized = tokenizer.tokenize(example)
            for word in tokenized:
                try:
                    BagOfWordsExamples[word] = BagOfWordsExamples[word] + 1
                except:
                    BagOfWordsExamples[word] = 1
        vector = np.zeros((100,),dtype='float32')
        count = 0
        for word in BagOfWordsExamples:
            vector = vector + np.array(fasttext_model[word])*BagOfWordsExamples[word]
            count = count + BagOfWordsExamples[word]
        SenseLabels.append(vector/count)
    return SenseLabels

def parser():
    """
    Parse the arguements
    """
    parser = argparse.ArgumentParser(description="Emoji Prediction")
    parser.add_argument("-i", "--img_dir", help="Directory containing test images",
        default="imgs", type=str)
    parser.add_argument("-w", "--weights_path", help="Path to weights file",
        default="resnet152_weights_tf.h5", type=str)

    args = parser.parse_args()
    return args

def get_coco_description(img_name):
    return coco_descriptions[img_name]

def cosine_product(vec1, vec2):
    mag1 = np.linalg.norm(vec1, axis=-1)
    mag2 = np.linalg.norm(vec2, axis=-1)

    return np.abs(np.dot(vec1, vec2)) / (mag1 * mag2)

def emoji_embeddings():
    with open('emojis.json', 'r') as f:
        parsed = json.load(f)

    SenseLabels = [] 
    senses = ['nouns','adjectives','verbs']
    for x in range(len(parsed)):
        surfaceforms = []
        for sense in senses:
            for i in range(0,len(parsed[x]['senses'][sense])):
                for babelnetid in parsed[x]['senses'][sense][i]:
                    for word in parsed[x]['senses'][sense][i][babelnetid]['surfaceforms']:
                        surfaceforms.append(word)
        temp = surfaceforms + parsed[x]['keywords']
        vector = np.zeros((100,),dtype='float32')
        for word in temp:
            vector = vector + np.array(fasttext_model[word])
        SenseLabels.append(vector/len(temp))
    return SenseLabels

def get_class_embeddings():
    embeddings = []
    with open('imagenet_class_index.json', 'r') as f:
        class_names = json.load(f)
    for k in range(1000):
        embeddings.append(fasttext_model[class_names[str(k)][1]])
    return np.array(embeddings).astype(np.float32)

def predict(model, img_path):
    preds = model.predict(os.path.join('imgs', img_path))
    class_embeddings = get_class_embeddings()
    sense_embeddings = emoji_embeddings()
    definition_embeddings = sense_definitions()
    name_embeddings = emoji_names()

    img_embedding = np.dot(preds, class_embeddings)
    description = get_coco_description(img_path)

    BagOfWords = {}
    for sentence in description:
        tokenized = tokenizer.tokenize(sentence)
        for key in tokenized:
            try:
                BagOfWords[key] = BagOfWords[key] + 1
            except:
                BagOfWords[key] = 1

    description_embedding = np.zeros((100),dtype='float32')
    count = 0
    for key in BagOfWords:
        description_embedding = description_embedding + np.array(fasttext_model[key])*BagOfWords[key]
        count = count + BagOfWords[key]
    description_embedding = description_embedding/count

    Final_image_embedding = (img_embedding + description_embedding)/2.0


    # Emoji scoring using emoji names
    similarity_names = []
    for i in range(len(name_embeddings)):
        prod = cosine_product(Final_image_embedding, name_embeddings[i])
        similarity_names.append(prod)
    similarity_names = np.array(similarity_names)
    top_k = []
    while len(top_k) <= 24:
        maximum = max(similarity_names)
        index = np.where(similarity_names == maximum)
        top_k.append(index[0][0])
        similarity_names[index] = -2
    np.save('imgs/{}_similarity_names.probs'.format(img_path), np.array(top_k))

    # Emoji scoring using emoji sense embeddings
    similarity = []
    for i in range(len(sense_embeddings)):
        prod = cosine_product(Final_image_embedding, sense_embeddings[i])
        similarity.append(prod)
    similarity = np.array(similarity)
    top_k = []
    while len(top_k) <= 24:
        maximum = max(similarity)
        index = np.where(similarity == maximum)
        top_k.append(index[0][0])
        similarity[index] = -2
    np.save('{}_similarity.probs'.format(img_path), np.array(top_k))

    # Emoji scoring using emoji sense definitions
    similarity_defns = []
    for i in range(len(sense_embeddings)):
        prod = cosine_product(Final_image_embedding, definition_embeddings[i])
        similarity_defns.append(prod)
    similarity_defns = np.array(similarity_defns)
    top_k = []
    while len(top_k) <= 24:
        maximum = max(similarity_defns)
        index = np.where(similarity_defns == maximum)
        top_k.append(index[0][0])
        similarity_defns[index] = -2
    np.save('{}_similarity_defns.probs'.format(img_path), np.array(top_k))

if __name__ == '__main__':
    args = parser()

    # Test pretrained model
    model = model()
    model.compile()
    model.set_weights(args.weights_path)

    for img in os.listdir(args.img_dir):
        predict(model, img)

