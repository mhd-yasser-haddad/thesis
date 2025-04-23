import unicodedata
from reference_parsing.utils.features_utils import scrub

class CanonicalFeature:
    def observe(self, token, alpha='', **opts):
        if not alpha:
            return 'BLANK'
        return self.canonize(alpha)

    def transliterate(self, string, form='NFKD'):
        # Normalize the string
        normalized_string = unicodedata.normalize(form, string)
        # Remove diacritical marks (accents)
        transliterated_string = ''.join(c for c in normalized_string if not unicodedata.combining(c))
        return transliterated_string

    def canonize(self, alpha):
        return scrub(self.transliterate(alpha)).lower()