import json

def read_prompt_question_from_json(file_path, is_dascore_format=False, add_prompt=False):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    data_questions = {}
    for i, item in enumerate(data):
        data_questions[i] = []
        if is_dascore_format:
            data_questions[i] = item['questions']
            if add_prompt:
                data_questions[i].append(f"is {item['prompt']}?")
            continue
        for question in item['questions']:
            data_questions[i].append(question["question"])
    
    return data_questions


def read_meta_data_question_from_json_openai(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    meta_data = {}
    for i, item in enumerate(data):
        type_questions = item['types']
        count_questions = item['counts'] # Dataset1
        # count_questions = item['count'] # Dataset2
        meta_data[i] = []
        for typeq, countq in zip(type_questions, count_questions):
            meta_data[i].append((typeq, countq))
    
    return meta_data


def read_image_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    boxes = {}
    for i, item in enumerate(list(data.values())):
        boxes[i] = []
        for box in item:
            boxes[i].append(box[0])
    return boxes


def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def write_json(file_path, data):
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
        
def read_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data

def read_prompt_question_from_api(parsed_input):
    data_questions = {}
    for i, item in enumerate(parsed_input):
        data_questions[i] = []
        for question in item['parsed_input']['questions']:
            data_questions[i].append(question)
    return data_questions


def read_meta_data_question_from_api(parsed_input):
    meta_data = {}
    for i, item in enumerate(parsed_input):
        type_questions = item['parsed_input']['type']
        meta_data[i] = []
        for typeq in type_questions:
            meta_data[i].append((typeq, 1 if typeq == "noun" else 2))
    return meta_data

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data