import regex as re

URI_REGEX = re.compile(
    r'^(?:http|https|ftp):\/\/(?:[a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,}(?:\/[a-zA-Z0-9\-._~:/?#[\]@!$&\'()*+,;%=]*)?$'
)
class LocatorFeature:
    def observe(self, token, **opts):
        if re.search(r'\b(DOI|doi|ISBN|Url|URL|PMCID|PMID|PMC\d+|PubMed)\b', token) or \
           re.search(r'10.\d{4,9}/[-._;()\/:A-Z0-9]+', token, re.IGNORECASE) or \
           bool(URI_REGEX.match(token)):
            return 'T'
        else:
            return 'F'
