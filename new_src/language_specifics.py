import re


'''
ENGLISH
'''
'''
CATEGORY = "category"
LIST = "list of"
REDIRECT = "redirect"
FILE = 'file'
IMAGE = 'image'
GALLERY = 'gallery'
LANG = ['lang']
AS_OF = 'as of'
CONVERT = ['convert', 'cvt']
TEMPLATE = 'template'
INFOBOX = 'infobox'
DASH = ['{{spaced ndash}}', '{{dash}}', '{{snd}}', '{{spnd}}', '{{sndash}}', '{{spndash}}', '&ndash;', '&mdash;']
RE_CATEGORIES = re.compile(r'\[\[Category:(.*?)\]\]', re.UNICODE)
RE_REMOVE_SECTIONS = re.compile(r'==\s*See also|==\s*References|==\s*Bibliography|==\s*Sources|==\s*Notes|==\s*Further Reading|==\s*External links',re.DOTALL | re.UNICODE)
CONVERT_TEMPLATE_SEPARATORS = {'-':'-','&ndash;':'-','and':' and ','and(-)':' and ','or':' or ','to':' to ','to(-)':' to ','to about':' to about ','+/-':' ± ','±':' ± ','&plusmn;':' ± ','+':' + ',',':', ',', and':', and ',', or':', or ','by':' by ','x':' by ','&times;':' by '}
RE_FILES = re.compile(r'\[\[(Image|File.*?)\]\]', re.UNICODE)
RE_CATEGORY_REDIRECT = re.compile(r'{{Category redirect\|([^}{]*)}}', re.DOTALL | re.UNICODE | re.MULTILINE)
# see https://en.wikipedia.org/wiki/Wikipedia:Namespace
IGNORED_NAMESPACES = [
    'wikipedia', 'category', 'file', 'portal', 'template',
    'mediaWiki', 'user', 'help', 'book', 'draft', 'wikiProject',
    'special', 'talk', 'image', 'module', 'gadget', 'timedtext', 'media'
]
RE_DISAMBIGUATIONS = '{{set index article}}|{{SIA}}|{{disambiguation\||\|disambiguation}}|{{disambiguation}}|{{disamb}}|{{disambig}}|{{disamb\||\|disamb}}|{{disambig\||\|disambig}}|{{dab\||\|dab}}|{{dab}}|{{disambiguation cleanup}}'
RE_HUMAN_DISAMBIGUATIONS = '{{hndis\||\|hndis}}|{{hndis}}|{{human name disambiguation}}|{{human name disambiguation\||\|human name disambiguation}}'
RE_GEO_DISAMBIGUATIONS = '{{place name disambiguation}}|{{geodis}}|{{geodis\||\|geodis}}'
RE_NUMBER_DISAMBIGUATIONS = '{{number disambiguation\||\|number disambiguation}}|{{numdab\||\|numdab}}|{{numberdis\||\|numberdis}}|{{numberdis}}|{{numdab}}|{{number disambiguation}}'
RE_STUB = 'stub}}'
GIVEN_NAMES = '{{given name}}', '[[Category:Given names]]', '[[Category:Masculine given names]]', '[[Category:Feminine given names]]'
SURNAMES = '{{surname}}', '[[Category:Surnames]]'
'''

'''
FRENCH
'''
CATEGORY = "catégorie"
LIST = "liste "
REDIRECT = "redirect"
FILE = 'fichier'
IMAGE = 'image'
TEMPLATE_QUOTE = 'citation'
LANG = ['langue']
AS_OF = 'as of'
CONVERT = ['conversion']
TEMPLATE = 'modèle'
INFOBOX = 'infobox'
DASH = ['{{spaced ndash}}', '{{dash}}', '{{snd}}', '{{spnd}}', '{{sndash}}', '{{spndash}}', '&ndash;', '&mdash;']
RE_CATEGORIES = re.compile(r'\[\[Catégorie:(.*?)\]\]', re.UNICODE)
RE_REMOVE_SECTIONS = re.compile(r'==\s*Voir aussi|==\s*Notes et références|==\s*Notes|==\s*Références|==\s*Bibliographie|==\s*Annexes|==\s*Liens externes',re.DOTALL | re.UNICODE)
CONVERT_TEMPLATE_SEPARATORS = {'-':'-','&ndash;':'-','and':' and ','and(-)':' and ','or':' or ','to':' to ','to(-)':' to ','to about':' to about ','+/-':' ± ','±':' ± ','&plusmn;':' ± ','+':' + ',',':', ',', and':', and ',', or':', or ','by':' by ','x':' by ','&times;':' by '}
RE_FILES = re.compile(r'\[\[(Image|Fichier.*?)\]\]', re.UNICODE)
RE_CATEGORY_REDIRECT = re.compile(r'{{catégorie redirect\|([^}{]*)}}', re.DOTALL | re.UNICODE | re.MULTILINE)
IGNORED_NAMESPACES = [
    'discussion', 'utilisateur', 'wikipédia', 'fichier', 'mediawiki',
    'modèle', 'aide', 'catégorie', 'portail', 'projet', 'référence',
    'module', 'sujet'
]
RE_DISAMBIGUATIONS = '{{homonymie\||\|homonymie}}|{{homonymie}}'
RE_HUMAN_DISAMBIGUATIONS = '{{homonymie de personnes\||\|homonymie de personnes}}|{{homonymie de personnes}}'
RE_HUMAN_DISAMBIGUATIONS = '{{hndis\||\|hndis}}|{{hndis}}|{{human name disambiguation}}|{{human name disambiguation\||\|human name disambiguation}}'
RE_GEO_DISAMBIGUATIONS = '{{toponymie}}|{{toponymie}}|{{toponymie\||\|toponymie}}'
RE_NUMBER_DISAMBIGUATIONS = '{{number disambiguation\||\|number disambiguation}}|{{numdab\||\|numdab}}|{{numberdis\||\|numberdis}}|{{numberdis}}|{{numdab}}|{{number disambiguation}}'
RE_STUB = 'ébauche}}'
GIVEN_NAMES = '{{prénom}}', '[[Catégorie:Prénom]]', '[[Catégorie:Prénom masculin]]', '[[Catégorie:Prénom féminin]]'
SURNAMES = '{{patronymie}}', '[[Catégorie:Patronyme]]'


'''
GERMAN
'''

'''
CATEGORY = "kategorie"
LIST = "liste "
REDIRECT = "weiterleitungshinweis"
FILE = 'datei'
IMAGE = 'bild'
TEMPLATE_QUOTE = 'zitat'
LANG = ['lang']
AS_OF = 'as of'
CONVERT = ['einheitenumrechnung']
TEMPLATE = 'vorlage'
INFOBOX = 'infobox'
DASH = ['{{spaced ndash}}', '{{dash}}', '{{snd}}', '{{spnd}}', '{{sndash}}', '{{spndash}}', '&ndash;', '&mdash;']
RE_CATEGORIES = re.compile(r'\[\[Kategorie:(.*?)\]\]', re.UNICODE)
RE_REMOVE_SECTIONS = re.compile(r'==\s*Siehe auch|==\s*Literatur|==\s*Weblinks|==\s*Anmerkungen|==\s*Quellen|==\s*Einzelnachweise',re.DOTALL | re.UNICODE)
CONVERT_TEMPLATE_SEPARATORS = {'-':'-','&ndash;':'-','and':' and ','and(-)':' and ','or':' or ','to':' to ','to(-)':' to ','to about':' to about ','+/-':' ± ','±':' ± ','&plusmn;':' ± ','+':' + ',',':', ',', and':', and ',', or':', or ','by':' by ','x':' by ','&times;':' by '}
RE_FILES = re.compile(r'\[\[(Bild|Datei.*?)\]\]', re.UNICODE)
RE_CATEGORY_REDIRECT = re.compile(r'{{Kategorie Weiterleitungshinweis\|([^}{]*)}}', re.DOTALL | re.UNICODE | re.MULTILINE)

# See https://de.wikipedia.org/wiki/Hilfe:Namensr%C3%A4ume
IGNORED_NAMESPACES = [
    'diskussion', 'benutzer', 'wikipedia', 'datei', 'mediawiki',
    'vorlage', 'hilfe', 'kategorie', 'portal', 'modul', 'gadget',
    'thema', 'spezial', 'medium'
]
RE_DISAMBIGUATIONS = '{{begriffsklärung\||\|begriffsklärung}}|{{begriffsklärung}}'
RE_HUMAN_DISAMBIGUATIONS = '{{hndis\||\|hndis}}|{{hndis}}|{{human name disambiguation}}|{{human name disambiguation\||\|human name disambiguation}}'
RE_GEO_DISAMBIGUATIONS = '{{place name disambiguation}}|{{geodis}}|{{geodis\||\|geodis}}'
RE_NUMBER_DISAMBIGUATIONS = '{{number disambiguation\||\|number disambiguation}}|{{numdab\||\|numdab}}|{{numberdis\||\|numberdis}}|{{numberdis}}|{{numdab}}|{{number disambiguation}}'
RE_STUB = 'lückenhaft}}'
GIVEN_NAMES = '{{given name}}', '[[Kategorie:Vorname]]', '[[Kategorie:Männlicher Vorname]]', '[[Kategorie:Weiblicher Vorname]]'
SURNAMES = '{{surname}}', '[[Kategorie:Familienname]]'
'''








