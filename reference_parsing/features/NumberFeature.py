import regex as re

class NumberFeature:
    def observe(self, token, **opts):
        if re.match(r'\d[\(:;]\d', token):
            return 'volume'
        elif re.match(r'^97[89](\p{Pd}?\d){10}$', token) or re.match(r'^\d(\p{Pd}?\d){9}$', token):
            return 'isbn'
        elif re.search(r'\b(1\d|20)\d\d\b', token):
            return 'year'
        elif re.match(r'^\d\d\d\d$', token):
            return 'quad'
        elif re.match(r'^\d\d\d$', token):
            return 'triple'
        elif re.match(r'^\d\d$', token):
            return 'double'
        elif re.match(r'^\d$', token):
            return 'single'
        elif re.match(r'^\d+$', token):
            return 'all'
        elif re.match(r'^\d+\p{Pd}+\d+$', token):
            return 'range'
        elif re.match(r'^\p{Lu}[\p{Lu}\p{Pd}\/]+\d+[,.:]?$', token):
            return 'idnum'
        elif re.match(r'\d\p{Alpha}{1,3}\b', token, re.IGNORECASE):
            return 'ordinal'
        elif re.search(r'\d', token):
            return 'numeric'
        elif re.match(r'^([IVXLDCM]+|[ivx]+)\b', token):
            return 'roman'
        else:
            return 'none'