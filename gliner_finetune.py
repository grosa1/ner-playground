import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset
import argparse
import json

# Define the base model
HF_GLINER_BASE_MODEL = "urchade/gliner_small"


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


if __name__ == "__main__":

    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()

    # Load the dataset
    with open('assets/train.json', 'r') as f:
        train_dataset = json.load(f)

    if os.path.exists('assets/eval.json'):
        with open('assets/eval.json', 'r') as f:
            eval_dataset = json.load(f)
    else:
        print("No eval dataset found, skipping")
        eval_dataset = []
    
    if os.path.exists('assets/test.json'):
        with open('assets/test.json', 'r') as f:
            test_dataset = json.load(f)
    else:
        print("No test dataset found, skipping")
        test_dataset = []

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = GLiNER.from_pretrained(HF_GLINER_BASE_MODEL)

    # use it for better performance, it mimics original implementation but it's less memory efficient
    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

    # Optional: compile model for faster training
    model.to(device)
    print("Model offload done")

    # calculate number of epochs
    num_steps = 500
    batch_size = 8
    data_size = len(train_dataset)
    num_batches = data_size // batch_size
    num_epochs = max(1, num_steps // num_batches)

    training_args = TrainingArguments(
        output_dir="models",
        learning_rate=5e-6,
        weight_decay=0.01,
        others_lr=1e-5,
        others_weight_decay=0.01,
        lr_scheduler_type="linear", #cosine
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        # evaluation_strategy="steps",
        evaluation_strategy="no",
        save_steps = 100,
        save_total_limit=10,
        dataloader_num_workers = 0,
        use_cpu = False,
        report_to="none",
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=test_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
