import os
import csv
# import pandas as pd

def Loader(folder_path):
    l = []
    files_in_folder = os.listdir(folder_path)
    
    for filename in files_in_folder:
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, "r", encoding="utf-8", newline="") as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\n')
            for row in reader:
                elements = row[0].split(',') if row else []
                word = elements[6]
                annotation = elements[7]
                l.append((word, {"entities" : [(int(elements[0]) , int(elements[1]) ,annotation)]}))
    
    # print(l)
    return l

# Loader("C:\\Users\\atrij\\OneDrive\\Desktop\\ML Internship\\dataset\\dataset\\train\\boxes_transcripts_labels")
                
            