import re
import json
from typing import List, Dict

def segment_to_sentence(text:str) -> List[str]:
    abbrev = {
        'M.': '§MONSIEUR§',
        'MM.': '§MESSIEURS§',
        'Dr.': '§DOCTEUR§'
    }

    temp_text = text
    for abbr, ph in abbrev.items():
        temp_text = temp_text.replace(abbr, ph)
    
    sentences = re.split(r'(?<=[.!?])(\s+)?(?=[A-ZÀÂÄÉÈÊËÎÏÔÙÛÜŸŒÇ])', temp_text)

    restored = []
    for s in sentences :
        if s is None or not s.strip():
            continue 
        for abbr, ph in abbrev.items():
            s = s.replace(ph, abbr)
        restored.append(s)
    
    # Filtrer les phrases trop courtes ?? si nécessaire ??
    sentences = [s.strip() for s in restored if len(s.strip())>10]

    return sentences 

def extract_sentence(questions: List[Dict]) -> List[Dict]:
    results = []
    for line in questions:
        uid = line["uid"]
        q = line["question"]

        if not q : 
            continue 

        if isinstance(q, list):
            question = ' '.join(q)

        sentences = segment_to_sentence(question)
        for i, sentence in enumerate(sentences, start=1):
            sentence_id = f"{uid}Q{i:02d}"
            results.append({
                "uid": uid,
                "id": sentence_id,
                "text": sentence 
            })
    
    return results 

def main():

    input_file = input("Entrez un fichier json de questions gouvermentales :").strip()
    output_file = "questions_qg.json"

    results = []
    with open(input_file, "r", encoding="utf8") as f:
        for line in f:
            results.append(json.loads(line))
    print (f"{len(results)} questions chargées.")

    sentences = extract_sentence(results)
    
    with open(output_file, "w", encoding="utf8") as f:
        for s in sentences : 
            json.dump(s, f, ensure_ascii=False)
            f.write('\n')

    print(f"{len(sentences)} phrases sauvegardé dans {output_file}")

    if sentences:
        print(f"\nExemple:")
        print(f"{sentences[0]}")


if __name__=="__main__":
    main()








