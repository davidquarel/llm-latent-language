LANG_TO_NAME = {
    'fr': 'Français',    # French
    'de': 'Deutsch',     # German
    'ru': 'Русский',     # Russian
    'en': 'English',     # English
    'zh': '中文',         # Chinese
    'es': 'Español',     # Spanish
    'ja': '日本語',       # Japanese
    'ko': '한국어',       # Korean
    'et': 'Eesti',       # Estonian
    'fi': 'Suomi',       # Finnish
    'nl': 'Nederlands',  # Dutch
    'hi': 'हिन्दी',       # Hindi
    'it': 'Italiano',    # Italian
}


# TRANSLATION_BANK = {
#             'day': {'zh': '日', 'en': 'day', 'fr': 'jour', 'de': 'Tag', 'ru': 'день'},
#             'man': {'zh': '男', 'en': 'man', 'fr': 'homme', 'de': 'Mann', 'ru': 'мужчина'},
#             'five': {'zh': '五', 'en': 'five', 'fr': 'cinq', 'de': 'fünf', 'ru': 'пять'},
#             'new': {'zh': '新', 'en': 'new', 'fr': 'nouveau', 'de': 'neu', 'ru': 'новый'}
#         }   

LANG_BANK = {
    'zh': ['日',   '男',     '五',   '新'  ],
    'en': ['day',  'man',    'five', 'new'],
    'fr': ['jour', 'homme',  'cinq', 'nouveau'],
    'de': ['Tag',  'Mann',   'fünf', 'neu'],
    'ru': ['день', 'мужчина','пять', 'новый'],
    'es': ['día',  'hombre', 'cinco','nuevo'],
    'ja': ['日',   '男',     '五',   '新'  ],
    'ko': ['날',   '남자',   '다섯', '새로운'],
    'et': ['päev', 'mees',   'viis', 'uus'],
    'fi': ['päivä','mies',   'viisi','uusi'],
    'nl': ['dag',  'man',    'vijf', 'nieuw'],
    'hi': ['दिन',  'आदमी',   'पांच', 'नया'],
    'it': ['giorno','uomo',  'cinque','nuovo']
}

MODELS_TOK_LEADING_SPACE = ["Llama-2-7b-hf", "Llama-2-7b", "Llama-2-7b-llama", "Llama-2-7b-llama-hf"]

LANGS_NO_SPACE = ['zh', 'ja', 'ko'] #languages that do not have spaces between words

LANGS = ['en', 'zh', 'fr', 'de', 'ru']