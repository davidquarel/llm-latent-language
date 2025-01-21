from ast import literal_eval
import warnings
import re
from src.llm import safe_tokenize
import torch
from tqdm.auto import tqdm
import csv

def parse_word_list(s):
    """
    Parses a string representation of a list of words.
    This function attempts to evaluate the string using `literal_eval` to convert it into a list.
    If that fails, it manually parses the string by removing the outer brackets and splitting
    the contents by commas. It also handles removing surrounding quotes from each word and
    replaces apostrophes within words.
    Args:
        s (str): A string representation of a list of words.
    Returns:
        list: A list of words parsed from the input string.
    Raises:
        UserWarning: If the string cannot be parsed using `literal_eval`.
    """
    # Remove the outer brackets and split by commas
    try:
        result = literal_eval(s)
        return result
    except:
        warnings.warn(f"Could not parse row: {s}")
        s = s.strip()[1:-1]
        items = re.split(r',\s*', s)
        
        result = []
        for item in items:
            # Remove surrounding quotes if present
            if (item.startswith("'") and item.endswith("'")) or (item.startswith('"') and item.endswith('"')):
                item = item[1:-1]
            # Handle apostrophes within words
            item = item.replace("'", "'")
            result.append(item)
    
        return result
    
def gen_lang_ids(df, model, langs):
    """
    Generate language-specific IDs for a given DataFrame using a specified model.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        model (object): The model used to generate IDs.
        langs (list): A list of language codes for which IDs need to be generated.

    Returns:
        dict: A dictionary where keys are language codes and values are the generated IDs.
    """
    id_bank = {}
    for lang in tqdm(langs, desc="Generating language IDs"):
        id_bank[lang] = gen_ids(df, model, lang)
    return id_bank

def gen_ids(df, model, lang):
    """
    Generate unique token IDs for words in a DataFrame column using a specified model and language.
    Args:
        df (pandas.DataFrame): DataFrame containing the data with columns for the primary language and its corresponding Claude language.
        model (transformers.PreTrainedModel): Pre-trained model used for tokenization.
        lang (str): The language code for the primary language column in the DataFrame.
    Returns:
        torch.Tensor: A padded tensor of unique token IDs for each word list in the DataFrame.
    """
    all_ids = []
    space_tok = safe_tokenize(" ", model).input_ids.item()
    for primary, word_list in df[[lang, f'claude_{lang}']].values:
        dest_words = [primary] + parse_word_list(word_list)
        padded_words = [" " + x for x in dest_words] + dest_words
            
        dest_ids = safe_tokenize(padded_words, model).input_ids[:, 0]
        dest_ids = dest_ids[dest_ids != space_tok]
        dest_ids = torch.unique(dest_ids)
        all_ids.append(dest_ids)
    pad_value = model.tokenizer.unk_token_id
    all_ids = torch.nn.utils.rnn.pad_sequence(all_ids, batch_first=True, padding_value=model.tokenizer.unk_token_id)
    return all_ids


def results_dict_to_csv(data, output_file, latent=False):
    # Ask for the output file name

    # Open the file in write mode
    with open(output_file, 'w', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)

        # Write the header row
        if latent:
            writer.writerow(['src_lang', 'dest_lang', 'latent_lang', 'avg_dest', 'sem95_error_dest', 'avg_latent', 'sem95_error_latent'])
        else:
            writer.writerow(['src_lang', 'dest_lang', 'latent_lang', 'avg', 'sem95_error'])

        # Iterate through the dictionary items
        for key, value in data.items():
            # Check if the key has 2 or 3 entries
            if len(key) == 2:
                src_lang, dest_lang = key
                latent_lang = ''
            else:
                src_lang, dest_lang, latent_lang = key

            if latent:
                avg_dest, sem95_error_dest, avg_latent, sem95_error_latent = value
                writer.writerow([src_lang, dest_lang, latent_lang, avg_dest, sem95_error_dest, avg_latent, sem95_error_latent])
            else:
                avg, sem95_error = value
                writer.writerow([src_lang, dest_lang, latent_lang, avg, sem95_error])

            # Write the row to the CSV file
            

    print(f"CSV file '{output_file}' has been created successfully.")

# Example usage:
# dict_to_csv(your_dictionary)