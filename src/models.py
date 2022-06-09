from re import A
import numpy as np
import torch
import torch.nn.functional as F

from typing import Iterable, List, Tuple
from torch import nn
from abc import ABC, abstractmethod
from datasets import Dataset
from dataclasses import dataclass
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertModel, BertForPreTraining
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import NextSentencePredictorOutput
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from enum import Enum
from scipy.special import softmax

from src.tokenization import get_tokenizer


bert_scaling_factor = 0.06


class LatentType(Enum):
    history = "history"
    intent = "intent"
    action = "action"

class ModelType(Enum):
    generator = "generator"
    retriever = "retriever"
    controller = "controller"



def _chunk_tensor(t: Iterable, chunk_size: int = 32) -> List[Iterable]:
        return [t[i : i + chunk_size] for i in range(0, len(t), chunk_size)]


def _truncate(input_ids, attention_mask):
        max_length = attention_mask.sum(-1).max().item()
        old_shape = input_ids.size()
        input_ids = input_ids.reshape([-1, old_shape[-1]])[:, :max_length]
        attention_mask = attention_mask.reshape([-1, old_shape[-1]])[:, :max_length]
        new_shape = list(old_shape[:-1]) + [max_length]
        input_ids = input_ids.reshape(new_shape)
        attention_mask = attention_mask.reshape(new_shape)
        
        return input_ids, attention_mask


def get_embedding(model: PreTrainedModel, input_ids: torch.tensor, attention_mask: torch.tensor = None):
    return model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :] * bert_scaling_factor


class Controller(BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, 3)

    def forward(
        self,
        input_ids = None,  # Just needed for Trainer to work, but we use query_input_ids
        attention_mask = None,
        query_input_ids = None,
        query_attention_mask = None,
        soft_labels = None
    ):

        embed = get_embedding(self.bert, query_input_ids, query_attention_mask)
        logits = self.linear(embed)

        loss = None
        if soft_labels is not None:
            loss = (- 1 * F.softmax(soft_labels, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(-1).mean(0) 

        return NextSentencePredictorOutput(loss=loss, logits=logits)

           
class Retriever(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.config.gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.tensor,
        attention_mask: torch.tensor,
        hist_input_ids: torch.tensor,
        hist_attention_mask: torch.tensor,
        act_input_ids: torch.tensor,
        act_attention_mask: torch.tensor,
        intent_input_ids: torch.tensor,
        intent_attention_mask: torch.tensor,
        labels: torch.tensor = None
    ):

        bsz = input_ids.shape[0]
        device = input_ids.device

        # Encode context
        ctx_embed = get_embedding(self.bert, input_ids, attention_mask)

        tgt_input_ids = [hist_input_ids, act_input_ids, intent_input_ids]
        tgt_attn_mask = [hist_attention_mask, act_attention_mask, intent_attention_mask]

        loss = 0
        loss_fn = nn.CrossEntropyLoss()
        labels = torch.arange(bsz).to(device)
        for t_ids, t_mask in zip(tgt_input_ids, tgt_attn_mask):
            
            # Encode targets
            tgt_embed = get_embedding(self.bert, t_ids, t_mask)

            scores = torch.matmul(ctx_embed, tgt_embed.T)        
            loss += loss_fn(scores, labels)

        return NextSentencePredictorOutput(loss=loss, logits=scores)  


class SubRetriever(ABC):
    @abstractmethod
    def get_top_n_docs(self, *args, **kwargs):
        raise NotImplementedError()

class IntentRetriever(SubRetriever):
    def __init__(self, retriever_encoder, tokenizer, device="cuda") -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.retriever_encoder = retriever_encoder.to(device)

        intents = ["navigate", "examine", "hoard", "interact", "other"]
        self.intent_tokens = self.tokenizer([i + ": " for i in intents], padding=True, return_tensors="np")
        with self.tokenizer.as_target_tokenizer():
            self.target_intent_tokens = self.tokenizer([i + ": " for i in intents], padding=True, return_tensors="np")
        self.intent_embed = self.build_index()

    def build_index(self):
        return get_embedding(
            self.retriever_encoder,
            torch.from_numpy(self.intent_tokens.input_ids).to(self.retriever_encoder.device),
            attention_mask=torch.from_numpy(self.intent_tokens.attention_mask).to(self.retriever_encoder.device)
        ).detach().cpu().numpy()

    def get_top_n_docs(self, query, n_docs, do_sample=False, *args, **kwargs):
        scores = np.sum(query * self.intent_embed, axis=-1)
        if do_sample:
            idxs = np.random.choice(np.arange(scores.shape[0]), size=n_docs, replace=True, p=softmax(scores))
        else:
            idxs = np.argsort(scores)[-n_docs:].tolist()
        samples = {
            "input_ids": self.intent_tokens.input_ids[idxs], 
            "attention_mask": self.intent_tokens.attention_mask[idxs],
            "target_input_ids": self.target_intent_tokens.input_ids[idxs],
            "target_attention_mask": self.target_intent_tokens.attention_mask[idxs]
        }

        return samples

        
class HistoryRetriever(SubRetriever):
    """Used for retrieving from both observation and action history"""
    def __init__(self, retriever_encoder: PreTrainedModel, game_history, device="cuda"):
        super().__init__()
        self.retriever_encoder = retriever_encoder.to(device)
        self.game2datasets = self.build_index(game_history)
        self.game_history = game_history
        
    def build_index(self, game_history, device="cuda"):
        """Builds a flat index for all observations in game history"""
        game2datasets = {}
        self.retriever_encoder.eval()
        with torch.no_grad():
            for game in game_history.game2hist.keys():
                input_ids = game_history.game2hist[game]["input_ids"]
                attention_mask = game_history.game2hist[game]["attention_mask"]

                input_ids = torch.tensor(input_ids).to(device)
                attention_mask = torch.tensor(attention_mask).to(device)

                input_ids = _chunk_tensor(input_ids)
                attention_mask = _chunk_tensor(attention_mask)

                embed = []
                for ids, mask in zip(input_ids, attention_mask):
                    embed.append(get_embedding(self.retriever_encoder, ids, mask))
                embed = torch.cat(embed).cpu().numpy()
                dataset = Dataset.from_dict({
                    "embeddings": embed,
                    "input_ids": np.array(game_history.game2hist[game]["input_ids"]),
                    "attention_mask": np.array(game_history.game2hist[game]["attention_mask"]),
                    "target_input_ids": np.array(game_history.game2hist[game]["target_input_ids"]),
                    "target_attention_mask": np.array(game_history.game2hist[game]["target_attention_mask"])
                })
                dataset.set_format(type="numpy")
                game2datasets[game] = dataset
        
        self.retriever_encoder.train()
        
        return game2datasets

    def get_top_n_docs(self, query: np.array, game: int, idx: int, n_docs: int, do_sample=False, *args, **kwargs):
        samples = self.game2datasets[game][:idx+1]
        embeds = samples["embeddings"]

        scores = np.sum(query * embeds, axis=-1)

        if do_sample:
            idxs = np.random.choice(np.arange(scores.shape[0]), size=n_docs, replace=True, p=softmax(scores)).tolist()
        else:
            idxs = np.argsort(scores)[-n_docs:].tolist()

        input_ids = samples["input_ids"][idxs] 
        attention_mask = samples["attention_mask"][idxs]
        target_input_ids = samples["target_input_ids"][idxs]
        target_attention_mask = samples["target_attention_mask"][idxs]

        samples = {
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
            "target_input_ids": target_input_ids,
            "target_attention_mask": target_attention_mask
        }

        return samples


@dataclass
class ModuleList:
    generator: PreTrainedModel = None
    retriever: PreTrainedModel = None
    controller: PreTrainedModel = None
    history_retriever: HistoryRetriever = None
    action_retriever: HistoryRetriever = None
    intent_retriever: IntentRetriever = None


def _get_retriever(encoder, game_history, action_history, tokenizer, latent_type: str = None):
    if latent_type == LatentType.action:
        return HistoryRetriever(encoder, action_history)
    elif latent_type == LatentType.history:
        return HistoryRetriever(encoder, game_history)
    elif latent_type == LatentType.intent:
        return IntentRetriever(encoder, tokenizer)
    else:
        raise ValueError("latent type not recognised.")

def _get_retrievers(encoder, game_history, action_history, tokenizer) -> Tuple:
    return map(lambda x: _get_retriever(encoder, game_history, action_history, tokenizer, x), list(LatentType))


def get_model(args, game_history=None, action_history=None):

    model_type = ModelType(args.model_type)

    if model_type in [ModelType.generator, ModelType.controller]:
        tokenizer = get_tokenizer(args)
        generator = GPT2LMHeadModel.from_pretrained(args.generator_path).to(args.device)
        retriever = BertModel.from_pretrained(args.retriever_path).to(args.device)
        retriever.eval()
        hist, act, intent = _get_retrievers(retriever, game_history, action_history, tokenizer)

        controller = Controller.from_pretrained(args.controller_path).to(args.device) if model_type == ModelType.controller else None

        return ModuleList(
            generator=generator,
            retriever=retriever,
            history_retriever=hist,
            action_retriever=act,
            intent_retriever=intent,
            controller=controller
        )    

    elif model_type == ModelType.retriever:
        model = Retriever.from_pretrained(args.model_path)
        return model

    else:
        raise NotImplementedError()

   
        


            


        
