from reference_parsing.config import DICT_FILE
from reference_parsing.embeddings.BaseReferenceEmbedding import BaseReferenceEmbedding

from reference_parsing.features.CapsFeature import CapsFeature
from reference_parsing.features.AffixFeature import AffixFeature
from reference_parsing.features.NumberFeature import NumberFeature
from reference_parsing.features.KeywordFeature import KeywordFeature
from reference_parsing.features.LocatorFeature import LocatorFeature
from reference_parsing.features.CategoryFeature import CategoryFeature
from reference_parsing.features.PositionFeature import PositionFeature
from reference_parsing.features.BracketsFeature import BracketsFeature
from reference_parsing.features.TerminalFeature import TerminalFeature
from reference_parsing.features.CanonicalFeature import CanonicalFeature
from reference_parsing.features.DictionaryFeature import DictionaryFeature
from reference_parsing.features.PunctuationFeature import PunctuationFeature
from reference_parsing.features.DictionaryFeature import parse_dictionary_file

from reference_parsing.utils.features_utils import scrub


class HandFeatureEmbedding(BaseReferenceEmbedding):
    def __init__(self, dictionary_data=parse_dictionary_file(DICT_FILE)):
        self.features = [
            CanonicalFeature(),
            CategoryFeature(index=[0, -1], strip=False),
            AffixFeature(size=2, prefix=True, suffix=False),
            AffixFeature(size=2, prefix=True, suffix=True),
            CapsFeature(),
            NumberFeature(),
            DictionaryFeature(dictionary=dictionary_data),
            KeywordFeature(),
            PositionFeature(idx='idx', seq='seq'),
            PunctuationFeature(),
            BracketsFeature(),
            TerminalFeature(),
            LocatorFeature()
        ]
    
    def get_token_embedding(self, token: str, alpha='', idx=None, seq=None):
        """
        For hand-engineered features, return a dictionary of observations.
        """
        observations = [
            feature.observe(token, alpha=alpha, idx=idx, seq=seq)
            for feature in self.features
        ]
        return observations
    
    def get_reference_embedding(self, tokens: list):
        observations = self.get_hand_features(tokens)
        return observations
    
    def get_hand_features(self, tokens: list):
        input_observations = []
        sequence = tokens
        for idx_value, token_value in enumerate(sequence):
            alpha_value = scrub(token_value)
            observations_raw = self.get_token_embedding(token_value, alpha=alpha_value, idx=idx_value, seq=sequence)
            observations_raw = [[item] if isinstance(item, str) else item for item in observations_raw]
            observations = sum(observations_raw, [])
            input_observations.append(observations)
        return input_observations
    
    def get_hand_feature_vocab(self):
        return {
            "<UNK>": 0,  # unknown tokens
            "none": 1, 
            "parens": 2,
            "square-brackets": 3,
            "angle": 4,
            "closing-paren": 5,
            "opening-paren": 6,
            "closing-square-bracket": 7,
            "opening-square-bracket": 8,
            "closing-angle": 9,
            "opening-angle": 10,
            "other": 11,
            "single": 12,
            "initial": 13,
            "caps": 14,
            "lower": 15,
            # For CategoryFeature, assume the outputs are exactly one of:
            "Lu": 16,
            "Ll": 17,
            "Lm": 18,
            "L": 19,
            "M": 20,
            "N": 21,
            "Pc": 22,
            "Pd": 23,
            "Ps": 24,
            "Pe": 25,
            "Pi": 26,
            "Pf": 27,
            "P": 28,
            "S": 29,
            "Zl": 30,
            "Zp": 31,
            "Z": 32,
            "C": 33,
            # For DictionaryFeature: we assume it returns a 4-tuple of "T" or "F"
            "T_F_F_F": 34,  # example composite output; you'll need to combine the four values consistently
            # For KeywordFeature:
            "editor": 35,
            "author": 36,
            "translator": 37,
            "thesis": 38,
            "proceedings": 39,
            "journal": 40,
            "in": 41,
            "and": 42,
            "etal": 43,
            "page": 44,
            "volume": 45,
            "series": 46,
            "patent": 47,
            "report": 48,
            "edition": 49,
            "date": 50,
            "locator": 51,
            "pubmed": 52,
            "arxiv": 53,
            "accessed": 54,
            "roman": 55,
            "none_keyword": 56,  # in case keyword returns none
            # For LocatorFeature (binary "T" or "F")
            "T": 57,
            "F": 58,
            # For NumberFeature:
            "volume_num": 59,
            "isbn": 60,
            "year": 61,
            "quad": 62,
            "triple": 63,
            "double": 64,
            "single_num": 65,
            "all": 66,
            "range": 67,
            "idnum": 68,
            "ordinal": 69,
            "numeric": 70,
            "roman_num": 71,
            "none_num": 72,
            # For PositionFeature:
            "only": 73,
            "first": 74,
            "last": 75,
            "0": 76,
            "1": 77,
            "2": 78,
            "3": 79,
            "4": 80,
            "5": 81,
            "6": 82,
            "7": 83,
            "8": 84,
            "9": 85,
            "10": 86,
            # For PunctuationFeature:
            "colon": 87,
            "hyphen": 88,
            "period": 89,
            "amp": 90,
            # For TerminalFeature:
            "strong": 91,
            "moderate": 92,
            "weak": 93,
            "none_terminal": 94
        } 
