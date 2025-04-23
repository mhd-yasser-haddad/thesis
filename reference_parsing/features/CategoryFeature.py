import regex as re
import sys

class CategoryFeature:
    def __init__(self, index=[0, -1], strip=False):
        self.index = index
        self.strip = strip

    def observe(self, token, **opts):
        chars = self.chars(token)
        if not chars:
            return ['none', 'none']
        return [self.categorize(chars[i]) for i in self.index]

    def chars(self, token):
        if self.strip:
            return list(token.strip())
        return list(token)

    def categorize(self, char):
        if re.match(r'\p{Lu}', char):
            return 'Lu'
        elif re.match(r'\p{Ll}', char):
            return 'Ll'
        elif re.match(r'\p{Lm}', char):
            return 'Lm'
        elif re.match(r'\p{L}', char):
            return 'L'
        elif re.match(r'\p{M}', char):
            return 'M'
        elif re.match(r'\p{N}', char):
            return 'N'
        elif re.match(r'\p{Pc}', char):
            return 'Pc'
        elif re.match(r'\p{Pd}', char):
            return 'Pd'
        elif re.match(r'\p{Ps}', char):
            return 'Ps'
        elif re.match(r'\p{Pe}', char):
            return 'Pe'
        elif re.match(r'\p{Pi}', char):
            return 'Pi'
        elif re.match(r'\p{Pf}', char):
            return 'Pf'
        elif re.match(r'\p{P}', char):
            return 'P'
        elif re.match(r'\p{S}', char):
            return 'S'
        elif re.match(r'\p{Zl}', char):
            return 'Zl'
        elif re.match(r'\p{Zp}', char):
            return 'Zp'
        elif re.match(r'\p{Z}', char):
            return 'Z'
        elif re.match(r'\p{C}', char):
            return 'C'
        else:
            return 'none'