import regex as re

class KeywordFeature:
    def observe(self, token, alpha='', **opts):
        if token == '&':
            return 'and'
        keywords = {
            'editor': [
                r'^ed(s|itors?|ited?|iteurs?)?$', r'^(hg|hrsg|herausgeber)$', r'^(compilador)$', r'編'
            ],
            'author': [r'著', r'撰'],
            'translator': [
                r'^trans(l(ated|ators?|ation))?$', r'^übers(etz(t|ung))?$', r'^trad(uction|ucteurs?|uit)?$', r'譯'
            ],
            'thesis': [r'^(dissertation|thesis)$'],
            'proceedings': [r'^(proceedings|conference|meeting|transactions|communications|seminar|symposi(on|um))'],
            'journal': [r'^(Journal|Zeitschrift|Quarterly|Magazine?|Times|Rev(iew|vue)?|Bulletin|News|Week|Gazett[ea])'],
            'in': [r'^in$', r'收入'],
            'and': [r'^([AaUu]nd|y|e)$'],
            'etal': [r'^(etal|others)$'],
            'page': [r'^(pp?|pages?|S(eiten?)?|ff?)$'],
            'volume': [r'^(vol(ume)?s?|iss(ue)?|n[or]?|number|fasc(icle|icule)?|suppl(ement)?|j(ahrgan)?g|heft)$'],
            'series': [r'^(ser(ies?)?|reihe|[ck]oll(e[ck]tion))$'],
            'patent': [r'^patent$'],
            'report': [r'^report$'],
            'edition': [
                r'^(edn|edition|expanded|rev(ised)?|p?reprint(ed)?|illustrated)$',
                r'^editio|aucta$', r'^(aufl(age)?|\p{Alpha}*ausg(abe)?)$'
            ],
            'date': [
                r'^(nd|date|spring|s[uo]mmer|autumn|fall|winter|frühling|herbst)$',
                r'^(jan(uary?)?|feb(ruary?)?|mar(ch|z)?|apr(il)?|ma[yi]|jun[ei]?)$',
                r'^(jul[yi]?|aug(ust)?|sep(tember)?|o[ck]t(ober)?|nov(ember)?|de[cz](ember)?)$',
                r'年'
            ],
            'locator': [r'^(doi|url)$'],
            'pubmed': [r'^(pmid|pmcid)$'],
            'arxiv': [r'^(arxiv)$'],
            'accessed': [r'^(retrieved|retirado|accessed|ab(ruf|gerufen))$'],
            'roman': [r'^[ILXVMCD]{2,}$']
        }

        for category, patterns in keywords.items():
            for pattern in patterns:
                if re.match(pattern, alpha):
                    return category
        return 'none'