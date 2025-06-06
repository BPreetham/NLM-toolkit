from keybert import KeyBERT
from nltk.tokenize import sent_tokenize
import nltk
from graphviz import Digraph
import re
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction  # this line is required to load the OpenIE model

nltk.download('punkt')

# 1. Keyword Extraction
def extract_keywords(text, top_n=5):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return [kw[0] for kw in keywords]

# 2. Relationship Extraction using AllenNLP
def extract_relationships(summary):
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz"
    )
    output = predictor.predict(sentence=summary)
    triples = []
    for verb in output['verbs']:
        description = verb['description']
        # Parse format: [ARG0: X] [V: Y] [ARG1: Z]
        parts = re.findall(r"\[.*?: (.*?)\]", description)
        if len(parts) >= 2:
            triples.append((parts[0], parts[1], parts[2] if len(parts) > 2 else ""))
    return triples

# 3. Generate flowchart using Graphviz
def generate_flowchart_from_relationships(triples, filename='flowchart_relationships'):
    dot = Digraph(format='png')
    dot.attr(rankdir='LR')

    for subj, verb, obj in triples:
        subj_id = subj.replace(" ", "_")
        obj_id = obj.replace(" ", "_")
        dot.node(subj_id, subj)
        dot.node(obj_id, obj)
        dot.edge(subj_id, obj_id, label=verb)

    output_path = dot.render(filename=filename, cleanup=False)
    print(f"Flowchart saved as: {output_path}")

# 4. Main
if __name__ == "__main__": 
    text = input("Enter your summary:\n")

    print("\nExtracting Relationships...")
    relationships = extract_relationships(text)
    print("Extracted Triples:", relationships)

    generate_flowchart_from_relationships(relationships)
