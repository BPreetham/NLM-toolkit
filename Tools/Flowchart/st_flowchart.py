from keybert import KeyBERT
from nltk.tokenize import sent_tokenize
import nltk
from graphviz import Digraph
import re

nltk.download('punkt')

# 1. Keyword Extraction
def extract_keywords(text, top_n=5):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return [kw[0] for kw in keywords]

# 2. Extract relationships (basic chain format)
def extract_edges(keywords):
    edges = []
    for i in range(len(keywords) - 1):
        edges.append((keywords[i], keywords[i + 1]))
    return edges

# 3. Generate flowchart using Graphviz
def generate_flowchart_graphviz(keywords, edges, filename='flowchart_1'):
    dot = Digraph(format='png')
    dot.attr(rankdir='LR')  # left to right flow

    for kw in keywords:
        node_id = kw.replace(" ", "_")
        dot.node(node_id, kw)

    for src, dst in edges:
        dot.edge(src.replace(" ", "_"), dst.replace(" ", "_"))

    output_path = dot.render(filename=filename, cleanup=False)
    print(f"Flowchart saved as: {output_path}")

# 4. Main
if __name__ == "__main__":
    text = input("enter your summary")
    
    keywords = extract_keywords(text, top_n=5)
    print("Extracted Keywords:", keywords)

    edges = extract_edges(keywords)
    print("Edges:", edges)

    generate_flowchart_graphviz(keywords, edges)
