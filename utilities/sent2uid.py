import re
import json
import pandas as pd 
uid_path = "./transformed_output.fr.xml.csv"
iob_path = "./iob_output_last.json"
output_path = "iob_with_uid.json"

def norm(text: str) -> str : 
    return re.sub(r"\s+","",str(text).lower()) # supprime tous les espaces 

ref = pd.read_csv(uid_path, sep="\t", header=None, 
                  names=["uid","sent_id", "num", "sent"])

uid_dict = {norm(s): i for i, s in ref[["uid","sent"]].itertuples(index=False)}

with open(iob_path, "r", encoding="utf-8") as f1, \
    open(output_path, "w", encoding="utf-8") as f2:

    for line in f1 : 
        line = line.strip()
        if not line : 
            continue 
        
        item = json.loads(line)
        uid = uid_dict.get(norm(item.get("text","")))
        # Met uid en premier dans chaque objet json
        new_item = {"uid":uid}
        new_item.update(item)

        f2.write(json.dumps(new_item, ensure_ascii=False)+"\n")

print("L'uid pour chaque phrase est retrouvé dans ", output_path)