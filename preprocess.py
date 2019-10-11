import spacy
import re
import regex
import string

nlp = spacy.load("en_core_web_sm")

def get_entity(docs, words_dict={}):
    """Methods used for keeping record of entity and words
    """
    for ent in docs.ents:
        if ent.label_ in words_dict:
            value = words_dict[ent.label_]
            if ent.text not in value:
                value.append(ent.text)
                words_dict[ent.label_] = value
        else:
            words_dict[ent.label_] = [ent.text]

    return words_dict

def get_pos(docs, words_dict={}):
    """Method used for keeping record of POS and words
    """
    for token in docs:
        if token.pos_ in words_dict:
            value = words_dict[token.pos_]
            if token.text not in value:
                value.append(token.text)
                words_dict[token.pos_] = value
        else:
            words_dict[token.pos_] = [token.text]

    return words_dict

class TextPreprocessor:
    """Helps tokenization select out the right words. Ref https://fasttext.cc/docs/en/supervised-tutorial.html
    """
    def __init__(self):
        
        self.normalizeDoubleQuotesRegex = re.compile("[“”]")
        self.normalizeSingleQuotesRegex = re.compile("[’]")
        self.whitespaceRegex = re.compile("\\s+")
        self.endOfSentencePeriodRegex = regex.compile(
            "(?<![^\\s]{1,10}\\.[^\\s]{0,10})\\.( |$)")  # handles abbreviations like U.S.A
        self.normalizeDashesRegex = regex.compile("\p{Pd}");  # replace weird unicode dashes with the standard one
        self.filter = ''.join([chr(i) for i in range(1, 32)])
        self.cleanr = re.compile('<.*?>|&([a-z0-9]+|[0-9]{1,6}|#x[0-9a-f]{1,6});')
        
    def pre_process(self, text):
        if text == "" or text is None:
            return text
        cleaned_text = text.lower()
        cleaned_text = self.whitespaceRegex.sub(" ", cleaned_text)  # Normalize all whitespace to a single space " ".
        cleaned_text = self.normalizeDoubleQuotesRegex.sub(' ', cleaned_text)
        cleaned_text = self.normalizeSingleQuotesRegex.sub(" ", cleaned_text)
        cleaned_text = cleaned_text.replace(".. ", " ")
        cleaned_text = cleaned_text.replace(", ", " , ")
        cleaned_text=re.sub(r"\d", " ", cleaned_text)
        cleaned_text=cleaned_text.replace("-"," ")
        cleaned_text=cleaned_text.replace("`"," ")
        cleaned_text=cleaned_text.replace("â€'"," ")
        cleaned_text=cleaned_text.replace("\\"," ")
        cleaned_text=cleaned_text.replace(","," ")
        cleaned_text=cleaned_text.replace("/"," ")
        cleaned_text=cleaned_text.replace("<"," ")
        cleaned_text=cleaned_text.replace(">"," ")
        cleaned_text=cleaned_text.replace("+"," ")
        cleaned_text=cleaned_text.replace("-"," ")
        cleaned_text=cleaned_text.replace("'s"," ")
        cleaned_text=cleaned_text.replace("("," ")
        cleaned_text=cleaned_text.replace(")"," ")
        cleaned_text=cleaned_text.replace(" / "," ")
        cleaned_text=cleaned_text.replace(":"," ")
        cleaned_text=cleaned_text.replace(". "," ")
        cleaned_text = self.whitespaceRegex.sub(" ", cleaned_text)
        cleaned_text=cleaned_text.strip()
        cleaned_text=cleaned_text.rstrip(string.punctuation)
        return cleaned_text