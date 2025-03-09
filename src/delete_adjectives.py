def find_adjectives_and_delete(question, colors, nlp):
    doc = nlp(question)
    new_tokens = []
    i = 0
    while i < len(doc):
        token = doc[i]
        if token.pos_ == "DET" and i + 2 < len(doc) and doc[i+1].pos_ == "VERB" and doc[i+2].pos_ == "NOUN":
            new_tokens.append(token.text)  
            new_tokens.append(doc[i+2].text)
            i += 3  
            
        elif token.pos_ == "ADJ" and token.text not in colors: # not delete colors 
            i += 1
            
        else:
            new_tokens.append(token.text)
            i += 1
    
    modified_question = " ".join(new_tokens).replace(" ?", "?")
    return modified_question


def find_adjectives_and_delete_all(question, nlp):
    doc = nlp(question)
    new_tokens = []
    i = 0
    while i < len(doc):
        token = doc[i]
        if token.pos_ == "DET" and i + 2 < len(doc) and doc[i+1].pos_ == "VERB" and doc[i+2].pos_ == "NOUN":
            new_tokens.append(token.text)  
            new_tokens.append(doc[i+2].text)
            i += 3  
            
        elif token.pos_ == "ADJ": # delete adjectives all
            i += 1
            
        else:
            new_tokens.append(token.text)
            i += 1
    
    modified_question = " ".join(new_tokens).replace(" ?", "?")
    return modified_question