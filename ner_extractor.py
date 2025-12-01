from gliner2 import GLiNER2


# Load model once, use everywhere
ner_extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# extract named entities
def extract_entities(text: str):
    # here, you can adjust the custom list of extracted entities to your desires
    entities = ner_extractor.extract_entities(text, ["company", "person", "product", "location", "date"])
    return entities
