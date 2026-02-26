import json

input_file = "test.json"
output_file = "test_id_text.json"

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for line in fin:
        if not line.strip():
            continue

        item = json.loads(line)

        extracted = {
            "uid": item.get("uid"),
            "id": item.get("id"),
            "text": item.get("text")
        }

        fout.write(json.dumps(extracted, ensure_ascii=False) + "\n")

print("OK — fichier généré :", output_file)
