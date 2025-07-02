import os
from PIL import Image
import re as regex
import json
from tqdm import tqdm
from collections import Counter
from math import log2
import pandas as pd
from pathlib import Path


def check_image(t, image_path):
    b = False
    d = {"image path": image_path}
    try:
        with Image.open(image_path) as img:
            d['corrupted image'] = False
            width, height = img.size
            if width < 90 or height < 90:
                b = True
                d["width"] = width
                d['height'] = height
                #t.write(f"{image_path} HEIGHT: {height} WIDTH: {width}\n")
            else:
                d["width"] = None
                d['height'] = None

    except Exception as e:
        #t.write(f"{image_path} CAN'T OPEN\n")
        b = True
        d['corrupted image'] = True
    if b:
        t.append(d)

    
def check_text(text, t2, id, index, human_question):

    b = False
    d = {'id':id, 'index':index, 'human question':human_question.replace("\n", "\\n"), 'text':text.replace("\n", "\\n")}

    

    if len(text) ==0:
        b = True
        d['empty string'] = True
    else:
        d['empty string'] = False

    if regex.search(r"[^\x00-\x7F]", text):
        b = True
        d['contain non ASCII character'] = True
    else:
        d['contain non ASCII character'] = False

    if regex.fullmatch(r"[^a-zA-Z0-9]+", text):
        b = True
        d['contain only symbols'] = True
    else:
        d['contain only symbols'] = False

    if regex.fullmatch(r"\d+", text):
        b = True
        d['contain only digits'] = True
    else:
        d['contain only digits'] = False

    
    if regex.search(r"http[s]?://|www\.", text):
        b = True
        d['contain url'] = True
    else:
        d['contain url'] = False

    if regex.search(r"\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b", text):
        b = True
        d['contain email'] = True
    else:
        d['contain email'] = False
    
    if regex.search(r"[\U0001F600-\U0001F64F]", text):
        b = True
        d['contain emoji'] = True
    else:
        d['contain emoji'] = False
    

    '''# check if contain mostly consonant
    letters = regex.findall(r"[a-zA-Z]", text)
    consonants = regex.findall(r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]", text)
    if len(letters):
        if len(consonants) / len(letters) > 0.9:
            b = True
            d['contain mostly consonants'] = True
        else:
            d['contain mostly consonants'] = False
    else:
        d['contain mostly consonants'] = False

    # check if has high puntuation to word ratio
    punct_count = len(regex.findall(r"[^\w\s]", text))
    total_count = len(text)
    if len(text):
        if total_count > 0 and (punct_count / total_count) > 0.3:
            b = True
            d['high punctuation to word ratio'] = True
        else:
            d['high punctuation to word ratio'] = False
    else:
        d['high punctuation to word ratio'] = False'''
    
    if regex.search(r"(asdf|jkl|qwer|zxcv|mnbv|hjkl)", text.lower()):
        b = True
        d['look like keyboard smashing'] = True
    else:
        d['look like keyboard smashing'] = False

    
    '''if len(text):
        counter = Counter(text)
        probs = [count / len(text) for count in counter.values()]
        entropy = -sum(p * log2(p) for p in probs if p > 0)
        if entropy < 1.5:
            b = True
            d["contain repetitive character/word"] = True
        else:
            d["contain repetitive character/word"] = False
    else:
        d["contain repetitive character/word"] = False'''

    
    if b:
        #t2.write(f"{id} {index} {reason}:\n human_question: {human_question}\n{text}\n##################################\n\n") 
        
        t2.append(d)
        #print(t2)


def EDA(json_folder_path, image_folder_path):
    #t = open("./dataset/problematic_images.log", "w")
    #t2 = open("./dataset/problematic_text.log", "w")
    t = []
    t2 = []

    for folder_name in tqdm(os.listdir(json_folder_path)):
        folder_path = os.path.join(json_folder_path, folder_name)

        if os.path.isdir(folder_path): # only check folder

            for filename in tqdm(os.listdir(folder_path)): # for each json file in the folder
                file_path = os.path.join(folder_path, filename)

                with open(file_path, "r") as f: # open the file that is list of json
                    data = json.load(f)

                    # loop through each json in the list
                    for n in tqdm(data):
                        image_path = os.path.join(image_folder_path, n['id'])
                        check_image(t, image_path)

                        for index, conversation in enumerate(n["conversations"]):
                            check_text(conversation['value'], t2, n['id'], index, n["conversations"][0]['value'])
    if t:
        df = pd.DataFrame(t)
        df.to_csv("./dataset/problematic_images.csv", index=False)
        print('All problematic images saved at ./dataset/problematic_images.csv')

    else:
        print('all images look fine')
    if t2:
        df = pd.DataFrame(t2)
        df.to_csv("./dataset/problematic_text2.csv", index=False)
        print('All problematic text saved at ./dataset/problematic_text.csv')
    else:
        print('all text are clean')

def clean_text(path, json_folder_path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("file not found")
    
    

    #id_list = set(df['id']) # use set since the ids can not contains duplicate
                            # and set provide faster item search than list 
    id_list = {}
    for index, row in df.iterrows():
        id_list[row['id']] =  index
    
    clean_json = []

    for folder_name in tqdm(os.listdir(json_folder_path)):
        folder_path = os.path.join(json_folder_path, folder_name)

        if os.path.isdir(folder_path): # only check folder
            
            for filename in tqdm(os.listdir(folder_path)): # for each json file in the folder
                file_path = os.path.join(folder_path, filename)
                clean_json = []

                with open(file_path, "r") as f: # open the file that is list of json
                    data = json.load(f)

                    # loop through each json in the list
                    for n in tqdm(data):
                        if n['id'] in id_list:
                            if regex.search(r"[^\x00-\x7F]", n["conversations"][df['index'][id_list[n['id']]]]['value']):
                                
                                n["conversations"][df['index'][id_list[n['id']]]]['value'] = regex.sub(r'[^\x00-\x7F]+', '', n["conversations"][df['index'][id_list[n['id']]]]['value'])
                            else:
                                continue               
                        clean_json.append(n)
                if clean_json:
                    original_path = Path(file_path)
                    new_path = Path(*original_path.parts[3:])
                    Path("./dataset/ColonINST/Json-file-clean"+f'/{new_path}').parent.mkdir(parents=True, exist_ok=True)
                    with open("./dataset/ColonINST/Json-file-clean"+f'/{new_path}', "w") as f:
                        json.dump(clean_json, f, indent=2)
    


if __name__ == "__main__":
    EDA("./dataset/ColonINST/Json-file", "./dataset/ColonINST/Positive-images")
    #clean_text("./dataset/problematic_text.csv", "./dataset/ColonINST/Json-file")

