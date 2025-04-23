import regex as re

class TerminalFeature:
    def observe(self, token, **opts):
        if re.search(r'[\.\)\]]["\'”„’‚´«‘“`»」』\)\]]?$', token) or re.search(r',["\'”„’‚´«‘“`»」』\)\]]|["\'”„’‚´«‘“`»」』\)\]],$', token):
            return 'strong'
        elif re.search(r'[:"\'”„’‚´«‘“`»」』][,;:\p{Pd}!?.]?$', token):
            return 'moderate'
        elif re.search(r'[!?,;\p{Pd}]["\'”„’‚´«‘“`»」』]?$', token):
            return 'weak'
        else:
            return 'none'