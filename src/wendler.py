import pandas as pd
import os
from .constants import LANGS

def merge_all(data_path, how='inner'):
    df = pd.DataFrame()
    for lang in ['fr', 'de', 'ru', 'zh', 'ko']:
        path = os.path.join(data_path, lang, "clean.csv")
        new_df = pd.read_csv(path)[['word_original', 'word_translation']]
        new_df = new_df.rename(columns={'word_translation': lang})
        if df.empty:
            df = new_df
        else:
            df = df.merge(new_df, on='word_original', how=how)
    df = df.rename(columns={'word_original': 'en'})
    return df

def load_data(data_path, cfg):
    assert cfg.src_lang in LANGS
    assert cfg.dest_lang in LANGS

    keys = ['word_original', 'word_translation']

    src_df = pd.read_csv(os.path.join(data_path, cfg.src_lang, "clean.csv"))[keys]
    dest_df = pd.read_csv(os.path.join(data_path, cfg.dest_lang, "clean.csv"))[keys]

    src_df = src_df.rename(columns={'word_translation': cfg.src_lang})
    dest_df = dest_df.rename(columns={'word_translation': cfg.dest_lang})

    if cfg.src_lang == cfg.dest_lang:
        df = src_df
        if cfg.src_lang == 'en':
            df = df.drop(columns=['word_original'])
            return df
    else:
        df = src_df.merge(dest_df, on='word_original', how='inner')

        if cfg.src_lang == 'en' or cfg.dest_lang == 'en':
            df = df.drop(columns=['word_original'])
            return df

    df = df.rename(columns={'word_original': 'en'})
    return df