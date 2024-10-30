import copy
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset
import os, ast, json, spacy, argparse, pandas as pd
import random
import json
from datetime import datetime

RANDOM_SEED = 123
TOKENIZER_SUPPORTED_LANGUAGES = ["en", "fr", "it", "es", "nl"] 


def save_data(data, file_path, overwrite):
    """Save data to a file, handling overwriting based on user preference."""
    path = Path(file_path)
    assets_dir = path.parent
    if not assets_dir.exists():
        assets_dir.mkdir(parents=True, exist_ok=True)
    
    if not overwrite and path.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f"{path.stem}_{timestamp}{path.suffix}"
    
    with open(file_path, 'w') as f:
        json.dump(data, f)
    
    print(f"Data saved to {file_path}")


os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='optional arguments')
    parser.add_argument('--output_dir', '-o',
                        metavar='PATH',
                        dest='output_dir',
                        required=False,
                        default='results',
                        help='Path to the directory where checkpoints and results will be saved')

    required = parser.add_argument_group('required arguments')
    required.add_argument('--dataset', '-d',
                          metavar='NAME',
                          dest='dataset',
                          required=True,
                          help='Dataset name to be used for training')

    return parser


def format_dataset(df: pd.DataFrame) -> list:
    """
    Format the dataset to the required format
    """

    ft_dataset = []
    for r in df.to_dict(orient='records'):
        patterns = []
        try:
            for pm in ast.literal_eval(r["privacy_mask"]):
                patterns.append({"label": pm['label'], "pattern": pm['value'], "start": pm['start'], "end": pm['end']})
        except:
            print(json.dumps(r, indent=4))
            raise Exception()
        ft_dataset.append({
            "text": r['source_text'], 
            "patterns": patterns,
            "lang": r["language"]
            })

    return ft_dataset


def spacy_tokenizer(text, tokenizer_lang):
    """
    Tokenize the text using spacy tokenizer
    """
    return copy.deepcopy(tokenizer_lang)


def extract_annotations(text, patterns, tokenizer_lang):
    """
    Extract annotations from the text using the patterns
    """
    nlp = spacy_tokenizer(text, tokenizer_lang)
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)

    # Process the text
    doc = nlp(text)

    # Preparing the output format
    tokenized_text = [token.text for token in doc]
    ner = []
    for ent in doc.ents:
        start = ent.start
        end = ent.end-1  # Adjusting end index to be inclusive, not +1
        ner.append([start, end, ent.label_])

    return {"tokenized_text": tokenized_text, "ner": ner}


def train_eval_test_split(data_patterns, eval_split=0.2, test_split=0.0, train_file="train.json", eval_file="eval.json", test_file="test.json", overwrite=True, random_state=RANDOM_SEED):
    """Process data and split into training, validation, and testing datasets."""

    models_lang = dict()
    for lang in TOKENIZER_SUPPORTED_LANGUAGES:
        models_lang[lang] = spacy.blank(lang)

    training_data = list()
    for idx, d in enumerate(data_patterns):
        print(f"Processing {idx+1}/{len(data_patterns)}")
        data = extract_annotations(d["text"], d["patterns"], models_lang[d["lang"]])
        training_data.append(data)

    # Handle the data splitting
    if test_split > 0:
        train_val, test = train_test_split(training_data, test_size=test_split, random_state=random_state)
        save_data(test, Path('assets', test_file), overwrite)
    else:
        train_val = training_data

    eval_size = eval_split / (1 - test_split)  # Adjust eval size based on the remaining data
    train, val = train_test_split(train_val, test_size=eval_size, random_state=random_state)

    # Save the data
    save_data(train, Path('assets', train_file), overwrite)
    save_data(val, Path('assets', eval_file), overwrite)

    return training_data


if __name__ == "__main__":

    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()

    # Load the dataset
    df = pd.read_csv(args.dataset).sample(10000)  # TODO: remove sampling
    dataset = format_dataset(df)

    # Convert and split the data into training, validation, and testing datasets
    if not os.path.exists("assets"):
        train_eval_test_split(dataset, eval_split=0, test_split=0.1, overwrite=True)
        print("Data split and saved successfully")
