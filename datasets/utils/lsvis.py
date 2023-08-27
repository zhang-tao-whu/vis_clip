import json
import os

json_file = '../lsvis/train_instances.json'
with open(json_file, 'r') as f:
    json_file = json.load(f)

appeared_ids = []
for anno in json_file['annotations']:
    id = anno['category_id']
    appeared_ids.append(id)
appeared_ids = list(set(appeared_ids))

catrgories_ = []
for id in appeared_ids:
    category = json_file['categories'][id - 1]
    assert id == category['id']
    print(id, category)
    catrgories_.append(category)

def write_to_file(words, filename):
    with open(filename, 'w') as file:
        for i, word in enumerate(words, start=1):
            line = str(i) + ':' + word + '\n'
            file.write(line)

def get_word(name):
    if '_(' in name:
        parts = name.split('_(')
        assert ')' == parts[-1][-1]
        parts[-1] = parts[-1][:-1]
        assert len(parts) == 2
        word = parts[-1] + ' ' + parts[0]
    else:
        if '_' in name:
            parts = name.split('_')
            word = parts[0]
            for part in parts[1:]:
                word += ' '
                word += part
        else:
            word = name
    return word

words = []
for i in range(catrgories_[-1]['id']):
    if i + 1 in appeared_ids:
        word = get_word(json_file['categories'][i]['name'])
    else:
        word = 'invalid_class_id'
    words.append(word)

write_to_file(words, './lsvis_instance_with_prompt_eng.txt')

json_file['categories'] = catrgories_
with open('./train_instances_.json', 'w') as f:
    json.dump(json_file, f)

