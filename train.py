import torch
import random
import numpy as np
import os
import pickle

from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser, 
    Trainer,
    TrainerCallback, 
    TrainingArguments, 
    DataCollatorWithPadding
)
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from datasets import load_metric
from transformers import BertTokenizer

from src.dataset import get_dataset
from src.models import get_model, ModelType
from src.tokenization import get_tokenizer

os.environ["WANDB_DISABLED"] = "true"


@dataclass
class Args:
    # General parameters
    model_type: str = field(default="generator", metadata={"choices": ["retriever", "generator", "controller"]})
    task_type: str = field(default="clubfloyd", metadata={"choices": ["clubfloyd", "jericho"]})

    # Dataset parameters
    data_directory: str = field(default="calm/cleaned_corpora", metadata={"help": "path to ClubFloyd transcripts."})
    dataset_load_path: str = None #"clubfloyd_dataset_t5" #"clubfloyd_dataset"
    dataset_save_path: str = "clubfloyd"
    max_state_length: int = 256
    max_action_length: int = 8
    max_length: int = 312
    test_size = 0.1
    num_negatives = 9
    debug: bool = False

    generator_path: str = "distilgpt2"
    retriever_path: str = "huawei-noah/TinyBERT_General_4L_312D"
    controller_path: str = "huawei-noah/TinyBERT_General_4L_312D"

    generator_tokenizer: str = "gpt2"
    retriever_tokenizer: str = "bert-base-uncased"

    # Training parameters
    model_path: str = "huawei-noah/TinyBERT_General_4L_312D" # "reranker_sl/checkpoint-1698"  # "huawei-noah/TinyBERT_General_4L_312D"
    output_dir: str = "dummy"
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    epochs: int = 1
    bsz: int = 8
    seed: int = 27
    k: int = 30
    max_history_length: int = 64
    device: str = field(default="cuda", metadata={"choices": ["cuda", "cpu"]})
      

def set_random_seed(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


accuracy_metric = load_metric("accuracy")


def _compute_metrics(eval_prediction):
    pred = eval_prediction.predictions.argmax(-1)
    acc = accuracy_metric.compute(predictions=pred, references=eval_prediction.label_ids)

    return acc

def _augment_dataset_generator(samples, modules, args, tokenizer):
    queries = modules.retriever(
        torch.tensor(samples["query_input_ids"]).to(args.device),
        torch.tensor(samples["query_attention_mask"]).to(args.device)
    ).last_hidden_state[:, 0, :].cpu().numpy() * 0.06

    n_docs = 1

    input_ids = []
    labels = []

    for i, query in enumerate(queries):
        for sub_retriever in [modules.history_retriever, modules.intent_retriever, modules.action_retriever]:

            docs = sub_retriever.get_top_n_docs(
                    query=query[i:i+1], 
                    game=samples["games"][i], 
                    idx=samples["game_idxs"][i], 
                    n_docs=n_docs, 
                    knowledge_label=samples["knowledge_labels"][i],
                )

            doc_input_ids = [i for i in docs["target_input_ids"][0] if i != tokenizer.pad_token_id]
            doc_labels = [-100 for i in docs["target_input_ids"][0] if i != tokenizer.pad_token_id]

            input_ids.append(doc_input_ids + [tokenizer.eos_token_id] + samples["input_ids"][i])
            labels.append(doc_labels + [-100] + samples["labels"][i])

    # Padding
    max_length = args.max_length
    padded_input_ids = []
    padded_labels = []
    for ids, lab in zip(input_ids, labels):
        padded_input_ids.append(ids + [tokenizer.pad_token_id] * (max_length - len(ids)) if len(ids) < max_length else ids[-max_length:])
        padded_labels.append(lab + [-100] * (max_length - len(ids)) if len(ids) < max_length else lab[-max_length:])
    input_ids = padded_input_ids
    labels = padded_labels

    for lab, ids in zip(labels, input_ids):
        assert len(lab) == len(ids)
         
    return {
        "input_ids": input_ids,
        "labels": labels
    }


def _augment_dataset_controller(samples, modules, args, tokenizer):
    queries = modules.retriever(
        torch.tensor(samples["query_input_ids"]).to(args.device),
        torch.tensor(samples["query_attention_mask"]).to(args.device)
    ).last_hidden_state[:, 0, :].cpu().numpy() * 0.06

    n_docs = 1


    all_losses = []

    for i, query in enumerate(queries):
        losses = []

        for sub_retriever in [modules.history_retriever, modules.intent_retriever, modules.action_retriever]:

            docs = sub_retriever.get_top_n_docs(
                    query=query[i:i+1], 
                    game=samples["games"][i], 
                    idx=samples["game_idxs"][i], 
                    n_docs=n_docs, 
                    knowledge_label=samples["knowledge_labels"][i],
                )

            doc_input_ids = [i for i in docs["target_input_ids"][0] if i != tokenizer.pad_token_id]
            doc_labels = [-100 for i in docs["target_input_ids"][0] if i != tokenizer.pad_token_id]

            input_ids = doc_input_ids + [tokenizer.eos_token_id] + samples["input_ids"][i]
            labels = doc_labels + [-100] + samples["labels"][i]

            loss = modules.generator(
                torch.tensor([input_ids[-1]]).to(args.device),
                labels=torch.tensor([labels[-1]]).to(args.device)
            ).loss.item()
            losses.append(loss)
        
        all_losses.append(losses)
         
    return {
        "soft_labels": all_losses,
    }


def main():
    parser = HfArgumentParser([Args])
    args, = parser.parse_args_into_dataclasses()
    
    set_random_seed(args)

    data_collator = None
    compute_metrics = None

    model_type = ModelType(args.model_type)

    if model_type == ModelType.controller:
        dataset_dict, game_history, action_history = get_dataset(args)
        modules = get_model(args, game_history, action_history)
        tokenizer = get_tokenizer(args)

        model = modules.controller

        data_collator = DataCollatorWithPadding(tokenizer.generator)

        if args.dataset_save_path is None:
            dataset_dict.save_to_disk(args.dataset_save_path)
            pickle.dump(game_history, open("game_history.pkl", "wb"))
            pickle.dump(action_history, open("action_history.pkl", "wb"))

        # Augment dataset
        with torch.no_grad():
            dataset_dict = dataset_dict.map(
                lambda x: _augment_dataset_controller(x, modules, args, tokenizer.generator), 
                batched=True, 
                batch_size=args.bsz,
                )


    elif model_type == ModelType.generator:
        dataset_dict, game_history, action_history = get_dataset(args)
        modules = get_model(args, game_history, action_history)
        model = modules.generator
        
        tokenizer = get_tokenizer(args)

        model.resize_token_embeddings(len(tokenizer.generator))
        
        model.tokenizer = tokenizer

        if args.dataset_save_path is None:
            dataset_dict.save_to_disk("clubfloyd_rag")
            pickle.dump(game_history, open("game_history.pkl", "wb"))
            pickle.dump(action_history, open("action_history.pkl", "wb"))

        # Augment dataset
        with torch.no_grad():
            dataset_dict = dataset_dict.map(
                lambda x: _augment_dataset_generator(x, modules, args, tokenizer.generator), 
                batched=True, 
                batch_size=args.bsz,
                remove_columns=[col for col in dataset_dict.column_names["train"] if col not in ["input_ids", "labels"]]
                )

    elif model_type == ModelType.retriever:
        dataset_dict = get_dataset(args)
        model = get_model(args)
        data_collator = DataCollatorWithPadding(BertTokenizer.from_pretrained(args.model_path))
        compute_metrics = _compute_metrics

    else:
        raise NotImplementedError()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        eval_steps=5000,
        num_train_epochs=args.epochs,
        logging_strategy="steps",
        logging_steps=5000,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        learning_rate=args.learning_rate,
        save_strategy="epoch",
        warmup_steps=100
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    

    

    trainer.train()



if __name__ == "__main__":
    main()
