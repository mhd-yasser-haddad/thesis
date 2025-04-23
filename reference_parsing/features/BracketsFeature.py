import regex as re

class BracketsFeature:
    def observe(self, token, **opts):
        if re.fullmatch(r'^[^\(\[<>\)\]]+$', token):
            return 'none'
        elif re.fullmatch(r'^\(.*\)[,;:\p{Pd}\.]?$', token):
            return 'parens'
        elif re.fullmatch(r'^\[.*\][,;:\p{Pd}\.]?$', token):
            return 'square-brackets'
        elif re.fullmatch(r'^<.*>[,;:\p{Pd}\.]?$', token):
            return 'angle'
        elif re.search(r'\)[,;:\p{Pd}\.]?$', token):
            return 'closing-paren'
        elif re.match(r'^\(', token):
            return 'opening-paren'
        elif re.search(r'\][,;:\p{Pd}\.]?$', token):
            return 'closing-square-bracket'
        elif re.match(r'^\[', token):
            return 'opening-square-bracket'
        elif re.search(r'>[,;:\p{Pd}\.]?$', token):
            return 'closing-angle'
        elif re.match(r'^<', token):
            return 'opening-angle'
        else:
            return 'other'