import json 

input_path = "iob_output_last.json"
output_path = "iob_output_deduplicated.json"

id_unique = {}

with open(input_path, "r", encoding="utf8") as f:
    for line in f :
        entry = json.loads(line.strip())
        entry_id = entry["id"]

        if entry_id not in id_unique :
            id_unique[entry_id] = entry

with open(output_path, "w", encoding="utf8") as f : 
    for entry in id_unique.values():
        f.write(json.dumps(entry, ensure_ascii=False)+"\n")

print(f"{len(id_unique)} objets uniques après le filtrage.")
print(f"Fichier enregistré sous {output_path}")
    



    


