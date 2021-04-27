import json

for fold in ("train", "dev", "test"):
    with open("topic_"+fold+".json", "r") as data_file, open("topic_"+fold+".txt", "w") as text_file:
        texts = []
        for json_str in data_file:
            json_obj = json.loads(json_str)
            text = json_obj['text']
            texts.append(text.strip())
        text_file.write('\n'.join(texts))
