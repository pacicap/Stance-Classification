import json
import spacy

def anonymize_sentence(
    sentence, nlp, labels=None):
    """
    Anonymize a sentence by replacing specified entities with their labels.

    If no labels are specified, all recognized entities will be anonymized.

    :param sentence: The input sentence to be anonymized.
    :param nlp: The NLP model used for entity recognition.
    :param labels: A list of entity labels to be anonymized. If not provided, all entities will be anonymized.
    :return: Anonymized sentence.
    """
    # Default labels to anonymize if none are provided
    default_labels = [
        "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
        "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY",
        "QUANTITY", "ORDINAL", "CARDINAL"
    ]

    # If no labels are provided, use the default list
    if labels is None:
        labels = default_labels

    doc = nlp(sentence)
    ents = [e for e in doc.ents if e.label_ in labels]
    sorted_ents = sorted(ents, key=lambda e: e.start_char, reverse=True)
    for ent in sorted_ents:
        sentence = sentence[:ent.start_char] + "[" + ent.label_ + "]" + sentence[ent.end_char:]
    return sentence

def anonymize_json_file(input_file, output_file, nlp, labels=None):
    """
    Anonymize sentences from a JSON file and write the anonymized sentences to a new JSON file.

    :param input_file: The input JSON file containing sentences.
    :param output_file: The output JSON file to write anonymized sentences.
    :param nlp: The NLP model used for entity recognition.
    :param labels: A list of entity labels to be anonymized. If not provided, all entities will be anonymized.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    anonymized_data = []
    for entry in data:
        prompt = entry.get('prompt', '')
        completion = entry.get('completion', '')
        anonymized_prompt = anonymize_sentence(prompt, nlp, labels)
        anonymized_data.append({'prompt': anonymized_prompt, 'completion': completion})

    with open(output_file, 'w') as f:
        json.dump(anonymized_data, f, indent=2)


# Load the spaCy NLP model
nlp = spacy.load("en_core_web_sm")
anonymize_json_file("C:/Users/aband/OneDrive/Desktop/BTU_AI/Information_Extraction/project_3/Outputs/output_colab_best.json", "C:/Users/aband/OneDrive/Desktop/BTU_AI/Information_Extraction/project_3/Outputs/first_best.json", nlp, labels=['PERSON', 'GPE'])
