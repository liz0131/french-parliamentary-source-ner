
import re
import json 
import unidecode 
from html import unescape
from pathlib import Path
from typing import List, Dict, Optional

def clean_text(text: str) -> str:
    """ Nettoie des balises et décode des entités HTML """
    text = unescape(text)
    # text = re.sub(r'<[^>]+>', '', text) 
    text = re.sub(r'\([^)]*\)', ' ', text)  # Supprimer les annotations entre parathèses
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def find_deputy_name(text: str) -> Optional[str]:
    """ 
    Extrait le nom du député 
    grâce au regex 
    "La parole est à M.|Mme [NOM], pour" au début du texte
    """
    # text_start = text[:500]
    regex = r'La parole est à (?:le\s|la\s)?(((?:M\.|Mme)\s+)?[^\.]+?)(?:(,\s*(?:pour|au titre|au nom|député|députée|président|prédidente|que)|pour|au nom|\.)|(<strong>|<b>))'
    match = re.search(regex, text, re.IGNORECASE)

    if match:
        name=match.group(1).strip()
        return re.sub(r'<[^>]+>', '', name)
    return None

def find_minister_name(text: str) -> Optional[str]:
    # regex = r'((?:M\.|Mme)\s+[A-ZÉÈ][a-zéèêëàâäîïôùûüÿœç]+(?:\s+[A-ZÉÈ][a-zéèêëàâäîïôùûüÿœç]+)*),\s*(?:Première\s+|Premier\s+)?(?:ministre|secrétaire)'
    # regex = r'((?:M\.|Mme)\s+[^\.]+?),\s*(?:ministre\s+[^\.]+?\.)'
    # regex = r'((?:M\.|Mme)\s+[^\.]+?),\s*(?:Première\s+|Premier\s+)?(?:ministre|secrétaire)'
    regex = r'((?:M\.|Mme)\s+[^\.]+?),<\/strong><i>\s*(?:Première\s+|Premier\s+)?(?:ministre|secrétaire)'
    match = re.search(regex, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None 


def extract_question_response(text: str, uid: str, deputy_name: str, minister_name: Optional[str]) -> Dict:
    questions = []
    responses = []

    # Pour nettoyer les balises html
    def clean_html(content: str) -> str:
        return re.sub(r'<[^>]+>', '', content).strip()
    
    # new_speaker = r'(?:M\.|Mme)\s+(?:le\s+|la\s+|[A-ZÉÈÊËÀÂÄÎÏÔÙÛÜŸŒÇ][a-zéèêëàâäîïôùûüÿœç]+\s+[A-ZÉÈÊËÀÂÄÎÏÔÙÛÜŸŒÇ])'

    def accent_insensitive(name):
        """Rend la regex tolérante aux accents (avec ou sans)"""
        replacements = {
            'a': '[aàâäáã]', 'e': '[eéèêë]', 'i': '[iîïí]', 
            'o': '[oôöóõ]', 'u': '[uùûüú]', 'c': '[cç]', 'n': '[nñ]', 'y': '[yÿ]',
            'A': '[AÀÂÄÁÃ]', 'E': '[EÉÈÊË]', 'I': '[IÎÏÍ]',
            'O': '[OÔÖÓÕ]', 'U': '[UÙÛÜÚ]', 'C': '[CÇ]', 'N': '[NÑ]', 'Y': '[YŸ]',
            'é': '[eéèêë]', 'è': '[eéèêë]', 'ê': '[eéèêë]', 'ë': '[eéèêë]',
            'à': '[aàâäáã]', 'â': '[aàâäáã]', 'ä': '[aàâäáã]',
            'î': '[iîïí]', 'ï': '[iîïí]',
            'ô': '[oôöóõ]', 'ö': '[oôöóõ]',
            'ù': '[uùûüú]', 'û': '[uùûüú]', 'ü': '[uùûüú]',
            'ç': '[cç]', 'ÿ': '[yÿ]',
            'É': '[EÉÈÊË]', 'È': '[EÉÈÊË]', 'Ê': '[EÉÈÊË]', 'Ë': '[EÉÈÊË]',
            'À': '[AÀÂÄÁÃ]', 'Â': '[AÀÂÄÁÃ]', 'Ä': '[AÀÂÄÁÃ]',
            'Î': '[IÎÏÍ]', 'Ï': '[IÎÏÍ]',
            'Ô': '[OÔÖÓÕ]', 'Ö': '[OÔÖÓÕ]',
            'Ù': '[UÙÛÜÚ]', 'Û': '[UÙÛÜÚ]', 'Ü': '[UÙÛÜÚ]',
            'Ç': '[CÇ]', 'Ÿ': '[YŸ]'
        }
        result = ''
        for char in name:
            if char in replacements:
                result += replacements[char]
            else:
                result += re.escape(char)
        return result

    if deputy_name: 
        # deputy_regex = re.escape(deputy_name)+r'\.\s+(.+?)(?=(?:M\.|Mme)\s+)'
        # deputy_regex = re.escape(deputy_name)+r'\.\s+(.+?)(?=' + new_speaker + r'|\Z)'
        deputy_regex = r'(?:<strong>|<b>)'+accent_insensitive(deputy_name)+r'\.\s*(?:<\/strong>|<\/b>)\s*(.+?)(?=(?:<strong>|<b>)(?:M\.|Mme)\s)'
        match = re.findall(deputy_regex, text)
        questions.extend([clean_html(q) for q in match])
    
    if minister_name:
        # minister_regex = re.escape(minister_name)+r'\,\s*(?:ministre)[^\.]*\.\s+(.+?)(?=(?:M\.|Mme)\s+)'
        # minister_regex = re.escape(minister_name)+r'\,\s*(?:ministre)[^\.]*\.\s+(.+?)(?=' + new_speaker + r'|\Z)'
        minister_regex = r'(?:<strong>|<b>)'+re.escape(minister_name)+r'\,\s*(?:<\/strong>|<\/b>)<i>\s(?:ministre)[^\.]*\.\s+<\/i>(.+?)(?=(?:M\.|Mme)\s+)'
        match = re.findall(minister_regex, text)
        responses.extend([clean_html(q) for q in match])

    return {
        "uid": uid,
        "deputy": deputy_name,
        "minister": minister_name,
        "question": questions,
        "response": responses
    }
    
def process_xml_folder(xml_path: str) -> List[Dict]:
    """Traite un fichier XML qui contient une seule question."""
    path = Path(xml_path)
    xml_files = sorted(path.glob("*.xml"))
    
    print(f"{len(xml_files)} fichier(s) trouvé(s) dans {xml_path}\n")
    
    results = []
    
    for file in xml_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()

        uid_match = re.search(r'<uid>(.+?)</uid>', content)
        text_match = re.search(r'<texte>(.+?)</texte>', content)

        if not uid_match or not text_match:
            print(f"{file.name}: uid ou texte non trouvé")
            continue

        uid = uid_match.group(1) 
        text = clean_text(text_match.group(1))

        deputy_name = find_deputy_name(text)
        minister_name = find_minister_name(text)

        if not deputy_name:
            print(f"{uid}: nom du député non reconnu")
            continue

        result = extract_question_response(text, uid, deputy_name, minister_name)
        results.append(result)

    return results

def process_xml_file(xml_path: str) -> List[Dict]:
    """Traite un fichier XML qui contient plusieurs questions."""
    with open(xml_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = []
    
    # Trouver toutes les <question>...</question>
    questions = re.findall(r'<question[^>]*>(.+?)</question>', content, re.DOTALL)
    
    print(f"{len(questions)} question(s) trouvée(s) dans {xml_path}\n")
    
    if questions: 
        for content in questions:
            # Extraire uid et texte de chaque question
            uid_match = re.search(r'<uid>(.+?)</uid>', content)
            text_match = re.search(r'<texte>(.+?)</texte>', content)
        
            if not uid_match or not text_match:
                continue
        
            uid = uid_match.group(1)
            text = clean_text(text_match.group(1))
        
            # Trouver député et ministre
            deputy_name = find_deputy_name(text)
            minister_name = find_minister_name(text)
        
            if not deputy_name:
                print(f"{uid}: nom du député non reconnu")
                continue

            result = extract_question_response(text, uid, deputy_name, minister_name)
            results.append(result)
    
    return results


def main():
    """Fonction principale."""
    input_path = input("Entrez le chemin (dossier ou fichier.xml): ")
    path = Path(input_path)
    
    # Traiter selon le type d'input
    if path.is_dir():
        results = process_xml_folder(str(path))
    elif path.is_file():
        results = process_xml_file(str(path))
    else:
        print(f"{input_path} non trouvé")
        return None
    
    # Sauvegarder
    with open("pool_qg.json", 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    print(f"{len(results)} résultat(s) sauvegardé(s)")
  
if __name__ == "__main__":
    main()


# text = "&lt;/p&gt;&lt;p align='CENTER'&gt; POSITION DE LA FRANCE SUR LE RÉGIME CUBAIN &lt;a name=PG7&gt;&lt;/a&gt; &lt;/p&gt;&lt;br&gt;&lt;strong&gt;M. le président. &lt;/strong&gt;La parole est à M. Damien Abad, pour le groupe Les Républicains.&lt;br&gt;&lt;br&gt;&lt;strong&gt;M. Damien Abad. &lt;/strong&gt;Monsieur le Premier ministre, les déclarations tragi-comiques sur Fidel Castro et le régime de Cuba de celle qui est désormais votre ministre de l'écologie sont le symbole de la déliquescence d'un pouvoir socialiste à la dérive. Aller rendre hommage à Fidel Castro est déjà, en soi, une faute de goût qu'aucun autre pays européen que la France n'a commise, à l'exception de la Grèce.&lt;br&gt;&lt;br&gt;Mais au lieu de faire profil bas, madame la ministre de l'écologie, vous avez encensé le régime de l'un des derniers dictateurs staliniens, contre toute évidence et en toute indécence. &lt;i&gt;(Applaudissements sur plusieurs bancs du groupe Les Républicains et du groupe de l'Union des démocrates et indépendants.)&lt;/i&gt;&lt;br&gt;&lt;br&gt;&lt;strong&gt;M. Jean-Jacques Candelier.&lt;/strong&gt; C'est faux !&lt;br&gt;&lt;br&gt;&lt;strong&gt;M. Damien Abad. &lt;/strong&gt;Cette faute de goût s'est transformée en faute morale car vous avez trahi la mémoire des victimes de l'oppression du régime cubain. Cette faute morale, enfin, est aussi une faute politique, parce que vous avez abaissé la parole de la France en tenant des propos complaisants sur cette dictature.&lt;br&gt;&lt;br&gt;&lt;strong&gt;M. André Chassaigne.&lt;/strong&gt; Parlez-nous de Batista !&lt;br&gt;&lt;br&gt;&lt;strong&gt;M. Damien Abad. &lt;/strong&gt;Comment pouvez-vous prétendre qu'il y a de la désinformation sur la réalité des droits de l'homme à Cuba, alors qu'y règnent la censure et la répression ? Que dites-vous à ces centaines de prisonniers politiques ? Comment pouvez-vous déclarer que les Cubains ont une liberté religieuse et de conscience, alors même qu'Amnesty International dresse un bilan accablant sur les libertés fondamentales ? &lt;i&gt;(Applaudissements sur les bancs du groupe Les Républicains et plusieurs bancs du groupe de l'Union des démocrates et indépendants.)&lt;/i&gt;&lt;br&gt;&lt;br&gt;&lt;strong&gt;M. Pierre-Yves Le Borgn'.&lt;/strong&gt; Très bien.&lt;br&gt;&lt;br&gt;&lt;strong&gt;M. Guy Teissier.&lt;/strong&gt; C'est honteux !&lt;br&gt;&lt;br&gt;&lt;strong&gt;M. Jean-Jacques Candelier.&lt;/strong&gt; C'est faux !&lt;br&gt;&lt;br&gt;&lt;strong&gt;M. Damien Abad. &lt;/strong&gt;Madame la ministre, la castritude aiguë, ça se soigne ! &lt;i&gt;(Sourires sur les bancs du groupe Les Républicains.) &lt;/i&gt;Contrairement à ce que vous avez déclaré, ce n'est pas la polémique qui est déplacée, ce sont vos propos !&lt;br&gt;&lt;br&gt;Madame la ministre, vous avez tenu ces propos au nom du gouvernement Valls. Ma question est simple, monsieur le Premier ministre : la confirmation dans ses fonctions de Mme Royal signifie-t-elle que le Président de la République et vous-même approuvez ses déclarations scandaleuses, qui abîment encore un peu plus l'image de la France à l'international ? &lt;i&gt;(Applaudissements sur les bancs du groupe Les Républicains et du groupe de l'Union des démocrates et indépendants.)&lt;/i&gt;&lt;br&gt;&lt;br&gt;&lt;strong&gt;M. le président. &lt;/strong&gt;La parole est à Mme la ministre de l’environnement, de l’énergie et de la mer, chargée des relations internationales sur le climat. &lt;i&gt;(Huées sur les bancs du groupe Les Républicains.)&lt;/i&gt;&lt;br&gt;&lt;br&gt;On écoute la réponse de Mme la ministre !&lt;br&gt;&lt;br&gt;&lt;strong&gt;Plusieurs députés du groupe Les Républicains.&lt;/strong&gt; La question s'adressait au Premier ministre !&lt;br&gt;&lt;br&gt;&lt;strong&gt;Mme Ségolène Royal,&lt;/strong&gt;&lt;i&gt; ministre de l’environnement, de l’énergie et de la mer, chargé des relations internationales sur le climat. &lt;/i&gt;Je peux y répondre moi-même. Un peu de liberté d'expression, s'il vous plaît ! &lt;i&gt;(Vives exclamations sur les bancs du groupe Les Républicains et du groupe de l'Union des démocrates et indépendants.)&lt;/i&gt;&lt;br&gt;&lt;br&gt;Comme à Cuba ! &lt;i&gt;(Applaudissements sur plusieurs bancs du groupe socialiste, écologiste et républicain. – Exclamations persistantes sur les bancs du groupe Les Républicains et du groupe de l'Union des démocrates et indépendants.)&lt;/i&gt;&lt;br&gt;&lt;br&gt;&lt;strong&gt;M. le président.&lt;/strong&gt; Mes chers collègues, monsieur Cochet, écoutez donc la réponse de Mme la ministre !&lt;br&gt;&lt;br&gt;&lt;strong&gt;Mme Ségolène Royal,&lt;/strong&gt;&lt;i&gt; ministre. &lt;/i&gt;Monsieur le député, je vous remercie de votre question qui me donne l'occasion de répondre à cette polémique. Je n'ai pas voulu, ni pu y répondre jusqu'à présent, car je respecte le temps des funérailles auxquelles je me suis rendu et pour lesquelles de centaines de milliers de Cubains se sont rassemblés. &lt;i&gt;(Rires sur les bancs du groupe Les Républicains et du groupe de l'Union des démocrates et indépendants.)&lt;/i&gt;&lt;br&gt;&lt;br&gt;&lt;strong&gt;M. le président.&lt;/strong&gt; S'il vous plaît…&lt;br&gt;&lt;br&gt;&lt;strong&gt;Mme Ségolène Royal,&lt;/strong&gt;&lt;i&gt; ministre. &lt;/i&gt;Je ne vous cache pas que la violence de cette polémique m'a surprise, et qu'elle a blessé les Cubains. &lt;i&gt;(Exclamations sur les bancs du groupe Les Républicains et du groupe de l'Union des démocrates et indépendants.)&lt;/i&gt;&lt;br&gt;&lt;br&gt;&lt;strong&gt;M. Pierre Lellouche.&lt;/strong&gt; Ce n'est pas la violence de la polémique qui est la plus grave, c'est celle du régime de Fidel Castro !&lt;br&gt;&lt;br&gt;&lt;strong&gt;Mme Ségolène Royal,&lt;/strong&gt;&lt;i&gt; ministre. &lt;/i&gt;Je l'ai dit et je le répète : c'est aux historiens de faire le bilan de cette histoire, avec sa part d'ombre et sa part de lumière. Je me suis exprimée sur la situation actuelle d'un pays magnifique et courageux…&lt;br&gt;&lt;br&gt;&lt;strong&gt;M. Guy Teissier.&lt;/strong&gt; Et misérable !&lt;br&gt;&lt;br&gt;&lt;strong&gt;Mme Ségolène Royal,&lt;/strong&gt;&lt;i&gt; ministre. &lt;/i&gt;…qui est en train de s'ouvrir. Il reçoit 4 millions de visiteurs, de touristes, chaque année. Vous n'allez pas me dire que 4 millions de personnes se rendent dans une dictature, monsieur le député, ce n'est pas vrai ! &lt;i&gt;(Mêmes mouvements.)&lt;/i&gt;&lt;br&gt;&lt;br&gt;&lt;strong&gt;M. le président.&lt;/strong&gt; S'il vous plaît ! Monsieur Lellouche !&lt;br&gt;&lt;br&gt;&lt;strong&gt;M. Jean-Jacques Candelier.&lt;/strong&gt; C'est scandaleux !&lt;br&gt;&lt;br&gt;&lt;strong&gt;Mme Ségolène Royal,&lt;/strong&gt;&lt;i&gt; ministre. &lt;/i&gt;Nous devons aujourd'hui accompagner l'ouverture de ce pays, comme le font de grandes entreprises françaises, qui ont été consternées par cette polémique. Ou alors il faudra m'expliquer, monsieur le député, pourquoi vous n'agressez pas de la même façon le pape, qui s'est rendu deux fois à Cuba et a rencontré personnellement Fidel Castro, de même que ses deux prédécesseurs. &lt;i&gt;(Les exclamations couvrent la voix de la ministre.)&lt;/i&gt;&lt;br&gt;&lt;br&gt;L'ambassadeur de France à Cuba, nommé par Nicolas Sarkozy en 2010, a déclaré qu'il n'y a plus de condamné détenu pour motif politique à Cuba, et qu'il faut cesser de caricaturer cet État. Je vous en prie, construisons positivement l'avenir de ce pays, renforçons les relations entre la France, l’Union européenne et Cuba. Le 14 décembre prochain sera signé un accord de coopération entre l'Union européenne et Cuba et je m'en réjouis profondément. &lt;i&gt;(Huées sur les bancs du groupe Les Républicains et du groupe de l'Union des démocrates et indépendants.)&lt;/i&gt;&lt;br&gt;&lt;br&gt;&lt;strong&gt;M. le président.&lt;/strong&gt; S'il vous plaît ! Vous savez bien ce que provoque, chez ceux qui nous regardent, ce genre de réaction ! Vous avez pu poser votre question, Mme la ministre a pu y répondre. Chacun a pu mesurer le contenu et de la question et de la réponse. Ne vous en faites pas, les électeurs savent voir et entendre !&lt;br&gt; &lt;p&gt;"

text="""La parole est à M. Bertrand Pancher, pour le groupe Union des démocrates et indépendants.&lt;br/&gt;&lt;b&gt;M. Bertrand Pancher.&lt;/b&gt; Monsieur le ministre délégué chargé des relations avec le Parlement, la gravité des crises que nous traversons depuis plus de quatre ans appelle, de la part de tous, des solutions aussi urgentes que puissantes.&lt;br/&gt;S'il est un domaine sur lequel les députés du groupe UDI peuvent vous suivre, c'est bien celui de l'environnement car c'est un enjeu fondamental pour l'avenir du pays, qui nécessite de réunir l'ensemble des acteurs pour parvenir à des solutions innovantes, partagées et durables s'inscrivant dans le long terme.&lt;br/&gt;Nous avons été nombreux à soutenir le processus du Grenelle de l'environnement, initié par Jean-Louis Borloo. Cette belle machine à créer des consensus s'est imposée dans notre paysage.&lt;br/&gt;Nous nous sommes associés à la démarche de continuité que vous sembliez vouloir adopter à travers la Conférence environnementale et le prochain débat sur la transition énergétique, bien que le démantèlement du ministère, la baisse du crédit d'impôt développement durable et des éco-prêts à taux zéro, ainsi que le recul sur les grandes infrastructures ferroviaires nous inquiètent vivement.&lt;br/&gt;Plus grave, la semaine dernière nous avons examiné en séance publique la proposition de loi instaurant un bonus-malus énergétique dans des conditions qui n'étaient pas dignes des enjeux. En effet, la procédure d'urgence a été engagée sur un texte qui ne sera appliqué qu'en 2014 ; la commission du développement durable n'a pas été consultée ; les travaux ont débuté avant l'ouverture de la Conférence environnementale censée s'emparer de ces sujets ; la concertation a été inexistante, ce qu'un grand nombre d'organisations ont dénoncé ; aucune étude d'impact n'a été effectuée ; enfin, nous avons vu l'apparition surprise de cavaliers législatifs à deux heures trente du matin sans aucune concertation préalable.&lt;br/&gt;Que le Gouvernement piétine les droits du Parlement avec la bénédiction passive de la majorité est grave. Qu'il s'exonère de toute concertation sur des questions aussi essentielles est une faute.&lt;br/&gt;Vous avez rompu avec la méthode participative du Grenelle de l'environnement et nous avons bien compris que le grand débat sur la transition énergétique était clos avant d'avoir commencé.&lt;br/&gt;Monsieur le ministre, ma question est simple : quand allez-vous exiger de votre majorité qu'elle adopte une attitude plus respectueuse du travail parlementaire et de celui de la société civile ? &lt;i&gt;(Applaudissements sur les bancs des groupes UDI et sur divers bancs du groupe UMP.)&lt;br/&gt;&lt;/i&gt;&lt;b&gt;M. le président.&lt;/b&gt; La parole est à M. le ministre délégué chargé des relations avec le Parlement.&lt;br/&gt;&lt;b&gt;M. Alain Vidalies,&lt;/b&gt; &lt;i&gt;ministre délégué chargé des relations avec le Parlement. &lt;/i&gt;Monsieur le député, le Gouvernement est naturellement très attentif lorsqu'un parlementaire, surtout de l'opposition, se plaint des conditions dans lesquelles un texte aurait été examiné. Aussi, il s'est penché de près sur les incidents survenus la semaine dernière, à deux heures du matin, lors de l'examen de la proposition de loi instaurant une tarification progressive de l'énergie.&lt;br/&gt;Vous avez évoqué très rapidement évoqué le fond. Je le ferai tout aussi rapidement. Il s'agissait d'un débat important...&lt;br/&gt;&lt;b&gt;M. François Rochebloine.&lt;/b&gt; Engagé à deux heures et demie du matin !&lt;br/&gt;&lt;b&gt;M. Alain Vidalies,&lt;/b&gt; &lt;i&gt;ministre délégué. &lt;/i&gt;...puisque les amendements portaient sur deux points qui auraient pu faire consensus, me semble-t-il.&lt;br/&gt;&lt;b&gt;Mme Laure de La Raudière.&lt;/b&gt; Ces amendements étaient des cavaliers législatifs !&lt;br/&gt;&lt;b&gt;M. Alain Vidalies,&lt;/b&gt; &lt;i&gt;ministre délégué. &lt;/i&gt;La première question était de savoir comment limiter les normes en matière d'installation d'éoliennes en France, nombre d'élus se plaignant de cette situation. Le second point, à partir de cette simplification, c'était les initiatives que l'on pouvait prendre pour sauver une filière en danger qui représente tout de même 11 000 emplois en France.&lt;br/&gt;&lt;b&gt;M. Patrick Ollier.&lt;/b&gt; Mais ces amendements n'avaient rien à voir avec le texte !&lt;br/&gt;&lt;b&gt;M. Alain Vidalies,&lt;/b&gt; &lt;i&gt;ministre délégué. &lt;/i&gt;Il y avait donc là une nécessité politique. &lt;i&gt;(Applaudissements sur les bancs des groupes SRC et écologiste.)&lt;br/&gt;&lt;/i&gt;Mais votre question concerne la procédure. Vous permettrez au parlementaire expérimenté que j'ai été d'avoir regardé de près ce qui avait pu se passer.&lt;br/&gt;Vous avez indiqué dans la presse, et vous répétez aujourd'hui, que le Gouvernement aurait présenté, à deux heures du matin, des amendements qui n'auraient pas été examinés préalablement en commission, ce qui justifierait votre courroux. Je veux simplement, en toute objectivité, vous renvoyer au compte rendu de la réunion de la commission des affaires économiques qui s'est tenue le lundi 1er octobre 2012 à quinze heures trente sous la présidence de Mme Frédérique Massat - c'est la seule chose que le Gouvernement pouvait faire, puisqu'il n'est pas partie prenante dans le travail parlementaire à ce stade. Il y est précisé que l'ensemble des amendements du Gouvernement qui occasionneront l'incident survenu le jeudi suivant ont été examinés en commission. &lt;i&gt;(" C'est faux ! " sur les bancs du groupe UMP.) &lt;/i&gt;Vous n'y étiez pas, mais la réalité c'est que la procédure a été respectée. &lt;i&gt;(Vives protestations sur les bancs du groupe UMP.)&lt;br/&gt;&lt;/i&gt;&lt;b&gt;M. Daniel Fasquelle.&lt;/b&gt; Ce n'est pas vrai ! Il n'y a eu aucun débat en commission !&lt;br/&gt;"""
text = clean_text(text)
print(text)
deputy = find_deputy_name(text)
minister = find_minister_name(text)
print(deputy)
print(minister)
uid = "QU890"
parts = extract_question_response(text,uid,deputy, minister)
print(parts)

