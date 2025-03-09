import re

def parse_output(output_str):
    output = {
        'assertions': [],
        'questions': [],
        'entities':[],
        'type': [],
    }

    lines = output_str.split('\n')
    current_key = None

    for line in lines:
        line = line.strip().lower()
        if not line:
            continue

        if current_key is None and ('assertions:' not in line and 'decomposable-caption:' not in line and 'questions:' not in line):
            continue

        if 'assertions:' in line:
            current_key = 'assertions'
        elif 'decomposable-caption:' in line:
            current_key = 'decomposable-caption'
        elif 'questions:' in line:
            current_key = 'questions'
        elif 'entities:' in line:
            current_key = 'entities'
        elif 'type:' in line:
            current_key = 'type'
        elif 'count:' in line:
            current_key = 'count'
        elif current_key:
                point = re.search(r'\d+\.\s*(.+)', line)
                if point:
                    output[current_key].append(point.group(1))
    return output