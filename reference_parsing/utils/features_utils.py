import regex as re
import unicodedata

def scrub(string, blacklist=r'[^\w\p{Lm}]'):
    # Replace invalid characters with the Unicode replacement character
    string = ''.join(
        c if unicodedata.category(c) != 'Cn' else '\ufffd'
        for c in string
    )
    # Remove unwanted characters using the blacklist regex
    return re.sub(blacklist, '', string)

def tokenize(text):
    # Using the delimiter to split the text
    delimiter_pattern = re.compile(r'(\s|\u0020)+|([\uFF01-\uFF64]|。|、)')
    tokens = delimiter_pattern.split(text)
    tokens = [tok for tok in tokens if tok and not tok.isspace()]
    return tokens