import torch
import torch.nn.functional as F
import random
import numpy as np
import pickle

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from torch.utils.data import DataLoader
from collections import Counter
from statistics import mean

from src.dataset import get_dataset
from src.models import get_model, get_embedding
from src.tokenization import get_tokenizer


# From PARL.AI
def _prec_recall_f1_score(pred_items, gold_items):
        """
        Compute precision, recall and f1 given a set of gold and prediction items.
        :param pred_items: iterable of predicted values
        :param gold_items: iterable of gold values
        :return: tuple (p, r, f1) for precision, recall, f1
        """
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


@dataclass
class Args:
    # General parameters
    model_type: str = field(default="controller", metadata={"choices": ["retriever", "generator", "controller"]})
    task_type: str = field(default="jericho", metadata={"choices": ["jericho"]})

    # Dataset parameters
    data_directory: str = field(default="test_game.jsonl", metadata={"help": "path to Jericho transcripts."})
    dataset_load_path: str = None #"clubfloyd_dataset_t5" #"clubfloyd_dataset"
    dataset_save_path: str = None
    max_state_length: int = 64
    max_action_length: int = 8
    max_history_length: int = 64
    max_length: int = 200

    debug: bool = False

    generator_path: str = "distilgpt2"
    retriever_path: str = "huawei-noah/TinyBERT_General_4L_312D"
    controller_path: str = "huawei-noah/TinyBERT_General_4L_312D"
    gold_save_path: str = None
    f1_save_path: str = None

    generator_tokenizer: str = "gpt2"
    retriever_tokenizer: str = "bert-base-uncased"

    device: str = field(default="cuda", metadata={"choices": ["cuda", "cpu"]})

    # Eval parameters
    bsz: int = 8
    seed: int = 27
    k: int = 30


@dataclass
class GenerationArgs:
    do_sample: bool = True
    num_beams: int = 1
    top_k: int = None
    top_p: float = None



def set_random_seed(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def main():
    parser = HfArgumentParser([Args, GenerationArgs])
    args, gen_args = parser.parse_args_into_dataclasses()
    
    set_random_seed(args)

    torch.set_grad_enabled(False)

    test_dataset, game_history, action_history = get_dataset(args)
    test_dl = DataLoader(test_dataset, shuffle=False, batch_size=1)

    tokenizer = get_tokenizer(args)

    modules = get_model(args, game_history, action_history)
    modules.generator.eval()
    modules.generator.resize_token_embeddings(len(tokenizer.generator))
    modules.retriever.eval()
    modules.controller.eval()
    

    game2gold = {}
    game2f1 = {}
    all_f1 = []
    all_gold = []

    for j, batch in enumerate(test_dl):
        # Two step sampling procedure

        # Sample an action from the controller
        controller_logits = modules.controller(
            query_input_ids=torch.tensor([batch["query_input_ids"]]).to(args.device),
            query_attention_mask=torch.tensor([batch["query_attention_mask"]]).to(args.device)
        ).logits
        controller_probs = F.softmax(controller_logits, dim=-1)
        idxs = torch.multinomial(controller_probs, num_samples=args.k, replacement=True)

        zeros = torch.zeros(idxs.shape).to(args.device)
        hist_cnt, int_cnt, act_cnt = map(lambda x: int(zeros.masked_fill(idxs == x, 1).sum().item()), list(range(3)))
        cnts = [hist_cnt, int_cnt, act_cnt]
        sub_retrievers = [modules.history_retriever, modules.intent_retriever, modules.action_retriever]

        # Generate query with main retriever
        query = get_embedding(
            modules.retriever,
            torch.tensor([batch["query_input_ids"]]).to(args.device),
            torch.tensor([batch["query_attention_mask"]]).to(args.device)
        ).cpu().numpy()

        preds = []

        for cnt, sub_ret in zip(cnts, sub_retrievers):
            docs = sub_ret.get_top_n_docs(
                    query=query, 
                    game=batch["games"][0], 
                    idx=batch["game_idxs"].item(), 
                    n_docs=cnt, 
                    do_sample=True
            )

            # Build input_ids
            for i in range(cnt):
                # Remove padding from doc
                doc = [w for w in docs["target_input_ids"][i] if w != tokenizer.generator.pad_token_id]
                input_ids = torch.cat([
                    torch.tensor([doc]),
                    torch.tensor([[tokenizer.generator.eos_token_id]]),
                    torch.tensor([batch["input_ids"]])], dim=-1).to(args.device)
                outputs = modules.generator.generate(
                    input_ids=input_ids,
                    do_sample=gen_args.do_sample,
                    num_beams=gen_args.num_beams,
                    top_k=gen_args.top_k,
                    top_p=gen_args.top_p,
                    max_length=input_ids.shape[-1] + 10,
                    min_length=input_ids.shape[-1] + 2,
                    pad_token_id=tokenizer.generator.eos_token_id
                )
                pred = tokenizer.generator.decode(outputs[0, input_ids.shape[-1]:], skip_special_tokens=True)
                #remove trailing space
                if len(pred) > 0 and pred[0] == " ":
                    pred = pred[1:]
                preds.append(pred)    

        game = batch["games"][0]

        if game not in game2gold:
            game2gold[game] = []

        if game not in game2f1:
            game2f1[game] = []

        # F1 metric
        f1 = max([_prec_recall_f1_score(p.split(), batch["act"][0].split()) for p in preds])
        all_f1.append(f1)
        game2f1[game].append(f1)

        # Gold Recall
        gold_recall = 0
        for pred in preds:
            if pred == batch["act"][0]:
                gold_recall += 1
                break
        all_gold.append(gold_recall)
        game2gold[game].append(gold_recall)

        if j % 100 == 0:
            print("Examples preds: ", preds)
            print("Current gold recall: ", mean(all_gold))

    for game, valid_recall in game2gold.items():
        print(game, mean(valid_recall))

    print(mean(all_gold))

    for game, valid_recall in game2f1.items():
        print(game, mean(valid_recall))

    print(mean(all_f1))

    pickle.dump(all_gold, open(args.gold_save_path, "wb"))
    pickle.dump(all_f1, open(args.f1_save_path, "wb"))


if __name__ == "__main__":
    main()
