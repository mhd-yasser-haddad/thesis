import gzip

def parse_dictionary_file(filepath):
    dictionary = {'name': set(), 'place': set(), 'publisher': set(), 'journal': set()}
    current_tag = None

    with gzip.open(filepath, 'rt', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#!'):
                current_tag = line[2:].strip()  # Get the tag after #!
            elif current_tag in dictionary:
                token = line.lower()
                dictionary[current_tag].add(token)

    return dictionary

TAGS = ['name', 'place', 'publisher', 'journal']
class DictionaryFeature:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def observe(self, token, alpha='', **opts):
        if not alpha:
            return ['F', 'F', 'F', 'F']
        token_lower = alpha.lower()
        results = []
        for tag in TAGS:
            if token_lower in self.dictionary[tag]:
                results.append('T')
            else:
                results.append('F')
        return results
    
