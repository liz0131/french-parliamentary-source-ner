import re
import json
import pandas as pd

ref= "./transformed_output.fr.xml.csv"
file = "./iob_with_uid.json"
output = "iob_with_context.json"

punct = re.compile(r"^[\W_]+$", re.UNICODE)

def punct_only(sent: str) -> bool :
    """ Vérifie si toute la phrase se compose des ponctuations """
    if sent is None : 
        return True
    s = str(sent).strip()
    if not s : 
        return True 
    s = re.sub(r"\s+", "", s)
    return bool(punct.fullmatch(s))

def find_context(uid, num, uid_to_sent, direction, max_skip=5):
    """
    direction = -1, if prev
    direction = +1, if next
    """
    pos = num+direction 
    steps = 0

    while steps < max_skip : 
        sent = uid_to_sent.get((uid,pos))
        if sent is None :
            return None     # si la phrase centrale est déjà la frontière du document
        if not punct_only(sent):
            return sent     # on vérifie si la phrase de context est valide
        pos += direction    # sinon on saute la phrase tout en ponctuation 
        steps += 1
    
    return None

df = pd.read_csv(ref, sep="\t", header=None,
                 names=["uid","sent_id","num","sent"],
                 dtype=str, encoding="utf-8")

df["num"] = df["num"].astype(int)

uid_to_sent = {
    (uid, num): sent 
    for uid, num, sent in df[['uid','num','sent']].itertuples(index=False)
}

unk = 0 
with open(file,"r",encoding="utf-8") as f1, \
    open(output, "w", encoding="utf-8") as f2 :
    for line in f1 : 
        line = line.strip()
        if not line :
            continue 

        item = json.loads(line)
        uid = str(item.get("uid",""))
        sent_id = item.get("id")

        if uid == "" or sent_id is None :
            item["prev"] = None
            item["next"] = None
            unk += 1

        else : 
            num = int(sent_id)+1
            item["prev"] = find_context(uid, num, uid_to_sent, direction=-1)
            item["next"] = find_context(uid, num, uid_to_sent, direction=+1)
        
        f2.write(json.dumps(item, ensure_ascii=False)+"\n")

print("Fichier json avec contexte est prêt :", output )
print(unk, "phrase(s) sans contexte")