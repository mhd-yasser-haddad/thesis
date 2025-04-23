import regex as re

class CapsFeature:
    def observe(self, token, alpha='', **opts):
        if re.fullmatch(r'\p{Upper}', alpha):
            return 'single'
        elif re.match(r'\p{Upper}\p{Lower}', alpha):
            return 'initial'
        elif re.fullmatch(r'\p{Upper}+', alpha):
            return 'caps'
        elif re.fullmatch(r'\p{Lower}+', alpha):
            return 'lower'
        else:
            return 'other'