import json
import pandas as pd 
uid_path = "transformed_output.fr.xml.csv"
iob_path = "iob_output_deduplicated.json"
output_path = "iob_with_uid.json"

ref = pd.read_csv(uid_path, sep="\t", header=None, 
                  names=["uid","sent_id", "num", "sent"])
ref["num"] = ref["num"].astype(int)

num_to_uid = {num:uid for num, uid in ref[["num","uid"]].itertuples(index=False)}

unk= 0 
with open(iob_path, "r", encoding="utf-8") as f1, \
    open(output_path, "w", encoding="utf-8") as f2:

    for line in f1 : 
        line = line.strip()
        if not line : 
            continue 
        
        item = json.loads(line)
        item_id = item.get("id")
        if item_id is None : 
            uid = None
            unk += 1
        else : 
            num = int(item_id)+1 
            uid = num_to_uid.get(num)
            if uid is None : 
                unk += 1

        # Met uid en premier dans chaque objet json
        new_item = {"uid":uid}
        new_item.update(item)
        f2.write(json.dumps(new_item, ensure_ascii=False)+"\n")

print("L'uid pour chaque phrase est retrouvé dans ", output_path)
print("Nombre de phrases unconnues :", unk)