import spacy

from typing import List
from enum import Enum, auto

# Handcrafted intents
class Intent(Enum):
    navigate = auto()
    examine = auto()
    hoard = auto()
    interact = auto()
    other = auto()


motion = [
    "north",
    "east",
    "south",
    "west",
    "northeast",
    "northwest",
    "southeast",
    "southwest",
    "up",
    "down",
    "out",
    "enter",
    "exit"
]

nlp = spacy.load("en_core_web_sm")

def match_lists(list_a, list_b):
    for item in list_a:
        if item in list_b:
            return True
    
    return False

def get_intents(actions: List[str]):
    intents = []
    doc_actions = []
    for i, action in enumerate(actions):
        words = action.split()

        examine = ["look", "examine"]
        take = ["take", "get", "pick", "grab", "inventory", "wear"]
        
        # Explore
        if match_lists(motion, words):
            intents.append(Intent.navigate.value - 1)
        
        # Examine
        elif match_lists(examine, words):
            intents.append(Intent.examine.value - 1)
        
        # Take
        elif match_lists(take, words):
            intents.append(Intent.hoard.value - 1)
        
        else:
            doc_actions.append(action)
            intents.append(-1)  # Dummy value

    results = []
    for i, doc in enumerate(nlp.pipe(doc_actions)):
        pos = [word.pos_ for word in doc]
        verb = match_lists(pos, ["VERB", "PROPN"])
        noun = match_lists(pos, ["NOUN"])
        results.append(Intent.interact.value - 1) if verb and noun else results.append(Intent.other.value - 1)

        if i % 1000 == 0:
            print(i)
    
    for i, intent in enumerate(intents):
        if intent == -1:
            intents[i] = results.pop(0)

    return intents
