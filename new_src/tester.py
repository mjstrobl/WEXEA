import re

RE_TEMPLATE_1 = re.compile(r'{{([^}{]*)}}', re.DOTALL | re.UNICODE | re.MULTILINE)


def process_template(template_lower, template):
    tokens = template.split("|")
    print(tokens)
    new_tokens = []
    i = 0
    while i < len(tokens):
        if '[[' in tokens[i] and ']]' not in tokens[i] and i < len(tokens) - 1 and ']]' in tokens[i+1]:
            new_tokens.append(tokens[i] + '|' + tokens[i+1])
            i += 1
        else:
            new_tokens.append(tokens[i])
        i += 1

    tokens = new_tokens

    print(tokens)

    if template_lower.startswith("quote"):
        if len(tokens) == 1:
            return ""
        return '"' + tokens[1] + '"'
    elif template_lower.startswith("wikt-lang") or template_lower.startswith("lang"):
        for i in reversed(range(len(tokens))):
            if '=' not in tokens[i]:
                return tokens[i]
    elif template_lower.startswith('as of'):
        lc = False
        since = False
        alt = None
        df_US = False
        year = None
        month = None
        day = None
        pre = None
        post = None
        if len(tokens) == 1:
            return ""
        else:
            for i in range(1,len(tokens)):
                token = tokens[i]
                if token.strip().startswith('lc') and ('y' in token or ("lc=" in token and len(token) == 3)):
                    lc = True
                elif token.strip().startswith('df'):
                    df_US = True
                elif token.strip().startswith('since') and '=n' not in token and '= n' not in token:
                    since = True
                elif token.strip().startswith("alt"):
                    if '=' in token and len(token.split('=')[1].strip()) > 0:
                        alt = token.split("=")[1]
                elif ("pre=" in token and len(token) > 4) or ("pre =" in token and len(token) > 5):
                    pre = token.split("=")[1]
                elif ("post=" in token and len(token) > 5) or ("post =" in token and len(token) > 6):
                    post = token.split("=")[1]
                elif '=' in token:
                    other_param = token
                elif not year:
                    year = token
                elif not month and len(token) > 0:
                    month = process_month(token.lower())
                elif not day and len(token) > 0:
                    day = token

            if lc:
                if since:
                    replacement = "since "
                else:
                    replacement = "as of "
            else:
                if since:
                    replacement = "Since "
                else:
                    replacement = "As of "

            if alt:
                replacement = alt

            if pre:
                replacement += pre + " "

            if not alt:
                if year and not month and not day:
                    replacement += year
                elif year and month and not day:
                    replacement += month + " " + year
                elif year and month and day:
                    if df_US:
                        replacement += month + " " + day + ", " + year
                    else:
                        replacement += day + " " + month + " " + year

            if post:
                replacement += post + " "

            return replacement
    elif template_lower.startswith('convert') or template_lower.startswith('cvt'):
        if len(tokens) == 1:
            return ""
        else:
            first_value = None
            second_value = None
            separator = None
            unit = None

            for i in range(1, len(tokens)):
                token = tokens[i]
                if not first_value:
                    first_value = token
                elif token in CONVERT_TEMPLATE_SEPARATORS:
                    separator = CONVERT_TEMPLATE_SEPARATORS[token]
                elif separator and not second_value:
                    second_value = token
                else:
                    unit = token
                    break
            if not first_value or len(first_value) == 0 or not unit or len(unit) == 0:
                return ''
            replacement = first_value
            if separator and second_value:
                replacement += separator + second_value

            replacement += ' ' + unit
            return replacement

    return ""

text = "'''Alexander III of Macedon''' ({{lang-grc-gre|Αλέξανδρος Γʹ ὁ Μακεδών}}, {{Lang|grc-Latn|[[Alexander|Aléxandros]] III ho Makedȏn}}; 20/21 July 356 BC –"

while True:
    m = re.search(RE_TEMPLATE_1, text)
    if m == None:
        break

    start = m.span()[0]
    end = m.span()[1]
    template = m.group(1)
    template_lower = template.lower()
    text_to_keep = process_template(template_lower, template)
    text = text[:start] + text_to_keep + text[end:]


print(text)