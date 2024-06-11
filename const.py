
language_labels = {
    'zh': '中文',
    'en': 'English',
    'fr': 'Français',
    'de': 'Deutsch',
    'ru': 'Русский'
}
    #{'day': {'zh': '日', 'en': 'day', 'fr': 'jour', 'de': 'Tag', 'ru': 'день'},

all_translation_banks = {'gemma' :
                            {'water': {'zh': '水', 'en': 'water', 'fr': 'eau', 'de': 'Wasser', 'ru': 'вода'},
                            'man': {'zh': '男', 'en': 'man', 'fr': 'homme', 'de': 'Mann', 'ru': 'муж'},
                            'five': {'zh': '五', 'en': 'five', 'fr': 'cinq', 'de': 'fünf', 'ru': 'три'},
                            'new': {'zh': '新', 'en': 'village', 'fr': 'nouveau', 'de': 'neu', 'ru': 'пя'}},
                        'llama':
                            {'day': {'zh': '日', 'en': 'day', 'fr': 'jour', 'de': 'Tag', 'ru': 'день'},
                            'man': {'zh': '男', 'en': 'man', 'fr': 'homme', 'de': 'Mann', 'ru': 'муж'},
                            'five': {'zh': '五', 'en': 'five', 'fr': 'cinq', 'de': 'fünf', 'ru': 'три'},
                            'new': {'zh': '新', 'en': 'village', 'fr': 'nouveau', 'de': 'neu', 'ru': 'пя'}}
                        }

# just use the same bank for both
translation_bank = all_translation_banks['llama']               
    
lang2name = {'fr': 'Français', 'de': 'Deutsch', 'en': 'English', 'zh': '中文', 'ru': 'Русский'}