import regex as re

class PunctuationFeature:
    def observe(self, token, **opts):
        if re.fullmatch(r'^\p{^P}+$', token):
            return 'none'
        elif ':' in token:
            return 'colon'
        elif re.search(r'\p{Pd}', token):
            return 'hyphen'
        elif '.' in token:
            return 'period'
        elif '&' in token:
            return 'amp'
        else:
            return 'other'
