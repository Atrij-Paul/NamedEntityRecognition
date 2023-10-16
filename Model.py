import DataLoader
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin 
from spacy.training.example import Example

nlp2 = spacy.load("C:\\Users\\atrij\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\en_core_web_sm\\en_core_web_sm-3.7.0")

db = DocBin()

train_data = DataLoader.Loader("C:\\Users\\atrij\\OneDrive\\Desktop\\ML Internship\\dataset\\dataset\\train\\boxes_transcripts_labels")

# for text , annot in tqdm(train_data):
#     doc = nlp2.make_doc(text)
#     ents = []
#     for start , end , label in annot["entities"]:
#         if not isinstance(start, int) or not isinstance(end, int):
#                 print(f"Skipping entity due to non-integer start or end: {entity}")
#                 continue
#         span = doc.char_span(start , end , label = label , alignment_mode = "contract")
#         if span is None:
#             print("Skipping entity")
#         else:
#             ents.append(span)
#     doc.ents = ents
#     db.add(doc)
    
# db.to_disk("./output/train.spacy")

# def process_train_data(train_data):
#     examples = []
#     for text, annot in tqdm(train_data, desc="Processing examples"):
#         doc = nlp2.make_doc(text)
#         ents = []

#         for entity in annot.get("entities", []):
#             start, end, label = entity

#             # Check that start and end are integers
#             if not isinstance(start, int) or not isinstance(end, int):
#                 print(f"Skipping entity due to non-integer start or end: {entity}")
#                 continue

#             span = doc.char_span(start, end, label=label, alignment_mode="contract")
#             if span is None:
#                 print(f"Skipping entity: {entity}")
#             else:
#                 ents.append(span)

#         example = Example.from_dict(doc, {"entities": ents})
#         examples.append(example)

    # return examples
    
    
    
def process_train_data(train_data):
    examples = []
    for text, annot in tqdm(train_data, desc="Processing examples"):
        doc = nlp2.make_doc(text)
        ents = []

        # Check if "entities" key is present in annot and is a list
        if "entities" in annot and isinstance(annot["entities"], list):
            for entity in annot["entities"]:
                # Check that entity is a tuple with three elements
                if isinstance(entity, tuple) and len(entity) == 3:
                    start, end, label = entity

                    # Check that start and end are integers
                    if isinstance(start, int) and isinstance(end, int):
                        span = doc.char_span(start, end, label=label, alignment_mode="contract")
                        if span is not None:
                            ents.append(span)
                    else:
                        print(f"Skipping entity due to non-integer start or end: {entity}")
                else:
                    print(f"Skipping entity due to incorrect format: {entity}")

        example = Example.from_dict(doc, {"entities": ents})
        examples.append(example)

    return examples

training_examples = process_train_data(train_data)

nlp3 = spacy.blank("en")

# Fine-tune the NER model
for epoch in range(10):
    losses = {}
    for example in tqdm(training_examples, desc="Training epoch"):
        nlp3.update([example], drop=0.5, losses=losses)

    print(losses)

# Save the fine-tuned model to disk
nlp3.to_disk("C:\\Users\\atrij\\OneDrive\\Desktop\\ML Internship\\ModelBest.spacy")