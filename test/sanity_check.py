def sanity_check(cfg):
    prompt = gen_prompt(src_words = src_words,
                    dest_words = dest_words,
                    src_lang = cfg.src_lang, 
                    dest_lang = cfg.dest_lang,
                    num_examples=cfg.num_multi_shot)
    kv_cache = gen_kv_cache(prompt, model)
    suffixes = gen_common_suffixes(suffix_words,
                                    src_lang = cfg.src_lang,
                                    dest_lang = cfg.dest_lang)
    suffix_toks = raw_tokenize(suffixes, model)
    result = run_with_kv_cache(suffix_toks.input_ids, kv_cache, model)
    logits = eindex(result.logits, suffix_toks.indices, "batch [batch] vocab")
    translated = list(zip(df[cfg.src_lang][:10], model.tokenizer.convert_ids_to_tokens(logits.argmax(dim=-1))))
    
    print(tabulate(translated, headers=[f'{cfg.src_lang=}', f'{cfg.dest_lang=}']))