import os
import json
from collections import OrderedDict

def process_and_sort_file_content(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        processed_lines = []
        for line in lines:
            tokens = line.strip().split()
            processed_line = [[int(tokens[1]), int(tokens[2]), int(tokens[3]), int(tokens[4])], tokens[0], float(tokens[5])]
            processed_lines.append(processed_line)
        
        processed_lines.sort(key=lambda x: x[2], reverse=True)
        
    return processed_lines

files_dict = {}

def convert_to_dictionary(path_to_directory):
    for i, filename in enumerate(sorted(os.listdir(path_to_directory))):
        if filename.endswith('.txt'):
            file_path = os.path.join(path_to_directory, filename)
            
            sorted_content = process_and_sort_file_content(file_path)
            
            files_dict[i] = sorted_content
            
    sorted_files_dict = OrderedDict(sorted(files_dict.items()))

    return sorted_files_dict
