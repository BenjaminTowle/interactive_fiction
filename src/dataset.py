import pickle
import numpy as np
import random
import os
import jsonlines

from datasets import Dataset, DatasetDict, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from enum import Enum
from transformers import BertTokenizer

from src.tokenization import get_tokenizer
from src.intents import get_intents
from src.models import ModelType


class TaskType(Enum):
    clubfloyd = "clubfloyd"
    jericho = "jericho"


class GameHistory:
    """Container to store tokenized observations"""
    def __init__(self, max_length) -> None:
        self.max_length = max_length
        self.game2hist = {}
        self.keys = ["input_ids", "attention_mask", "target_input_ids", "target_attention_mask"]

    def push(
        self, 
        game: str, 
        input_ids, 
        attention_mask,
        target_input_ids = None,
        target_attention_mask = None
    ):
        target_input_ids = input_ids if target_input_ids is None else target_input_ids
        target_attention_mask = attention_mask if target_attention_mask is None else target_attention_mask

        keys = self.keys
        values = [input_ids, attention_mask, target_input_ids, target_attention_mask]

        if game not in self.game2hist:
            self.game2hist[game] = {key: [value] for key, value in zip(keys, values)}
        else: 
            for key, value in zip(keys, values):
                self.game2hist[game][key].append(value)

    def get_history(self, game, idx):
        game, idx = game.item(), idx.item()
        return {
            "input_ids": self.game2ids[game][:idx+1],
            "attention_mask": self.game2mask[game][:idx+1]
        }


def _split_into_states_actions(lines, sep: str, return_observations=False):
    """
    Splits unprocessed lines into two lists of states and actions, removing the overlap
    """
    states = []
    observations = []
    actions = []
    for line in lines:
        line = line.replace("[STATE]", sep).replace("[ACTION]", sep).split(sep)[1:]
        assert len(line) == 4, "line should consist of state, action, state, action"
        state = sep.join(line[:3])
        action = line[3]
        states.append(state)
        actions.append(action)
        observations.append(line[2])

    actions = [a.replace("\n", "") for a in actions]
    if return_observations:
        return states, observations, actions

    return states, actions


def _process_clubfloyd_retriever(files, tokenizer, args):
    dataset_dict = {}
    random.shuffle(files)
    idx = int(len(files)*args.test_size)
    train_files = files[:-idx]
    test_files = files[-idx:]

    k = 0                               
    for split, fs in [("train", train_files), ("test", test_files)]:

        all_states = []
        all_actions = []
        all_obs = []

        for i, file in enumerate(fs):
            with open(os.path.join(args.data_directory, file)) as f:
                lines = f.readlines()
                states, observations, actions = _split_into_states_actions(lines, tokenizer.sep_token, return_observations=True)

                if states == [] or observations == []:
                    continue
                
                tfidf = TfidfVectorizer()
                tfidf.fit(states + observations)

                # For each action find the most relevant previous observation
                obs_embed = tfidf.transform(observations)
                act_embed = tfidf.transform(actions)

                scores = linear_kernel(act_embed, obs_embed)
                mask = np.ones([len(actions), len(observations)])

                for a in range(len(actions)):
                    for o in range(len(observations)):
                        if o > a:
                            mask[a, o] = 0

                scores *= mask

                obs_targets = [observations[scores[j].argmax()] for j in range(len(actions))]

                all_states += states
                all_actions += actions
                all_obs += obs_targets

                k += 1
  
        states_inputs = tokenizer(all_states, max_length=args.max_state_length, truncation=True)
        actions_inputs = tokenizer(all_actions, max_length=args.max_action_length, truncation=True, padding="max_length")
        obs_inputs = tokenizer(all_obs, max_length=args.max_history_length, truncation=True, padding="max_length")

        intent_labels = get_intents(all_actions)
        intents = ["navigate", "examine", "hoard", "interact", "other"]
        intent_tokens = tokenizer([i + ": " for i in intents], padding=True, return_tensors="pt")
        intent_input_ids = [intent_tokens.input_ids[i] for i in intent_labels]
        intent_attention_mask = [intent_tokens.attention_mask[i] for i in intent_labels]

        dict_ = {
            "input_ids": states_inputs.input_ids, 
            "attention_mask": states_inputs.attention_mask,
            "hist_input_ids": obs_inputs.input_ids,
            "hist_attention_mask": obs_inputs.attention_mask,
            "intent_input_ids": intent_input_ids,
            "intent_attention_mask": intent_attention_mask,
            "act_input_ids": actions_inputs.input_ids,
            "act_attention_mask": actions_inputs.attention_mask
        }

        dataset = Dataset.from_dict(dict_)

        dataset_dict[split] = dataset

    return dataset_dict




def _split_dialog_into_states_actions(utterances, sep_token, pad_token):
    
    for _ in range(3):
        utterances.insert(0, pad_token)

    states = []
    observations = []
    actions = []
    
    for i in range(3, len(utterances)):
        states.append(sep_token.join(utterances[i-3:i]))
        observations.append(utterances[i-1])
        actions.append(utterances[i])

    return states, observations, actions
        
            

def _process_dailydialog_generator_controller(tokenizer, args) -> Dataset:
    dataset_dict = load_dataset("daily_dialog") 

    game_history = GameHistory(args.max_history_length)
    action_history = GameHistory(args.max_action_length)
    k = 0                               
    for split in ["train", "test"]:

        all_states = []
        all_query_states = []
        all_actions = []
        games = []
        game_idxs = []
        knowledge_labels = []
        intent_labels = []

        for i, dialog in enumerate(dataset_dict[split]):

            if args.debug:
                if i == 3:
                    break

            states, observations, actions = _split_into_states_actions(dialog["dialog"], tokenizer.generator.eos_token, tokenizer.generator.pad_token)
            query_states = _split_into_states_actions(dialog["dialog"], tokenizer.question_encoder.sep_token, tokenizer.question_encoder.pad_token)[0]

            if states == [] or observations == []:
                continue

            obs_tokens = tokenizer(observations, max_length=args.max_history_length, padding="max_length", truncation=True)
            with tokenizer.as_target_tokenizer():
                obs_target_tokens = tokenizer(observations, max_length=args.max_history_length, padding="max_length", truncation=True)
            act_tokens = tokenizer(actions, max_length=args.max_action_length, padding="max_length", truncation=True)
            with tokenizer.as_target_tokenizer():
                act_target_tokens = tokenizer(actions, max_length=args.max_history_length, padding="max_length", truncation=True) 
            for j in range(len(observations)):
                game_history.push(
                    game=k, 
                    input_ids=obs_tokens.input_ids[j], 
                    attention_mask=obs_tokens.attention_mask[j],
                    target_input_ids=obs_target_tokens.input_ids[j],
                    target_attention_mask=obs_target_tokens.attention_mask[j]
                )
                action_history.push(
                    game=k,
                    input_ids=act_tokens.input_ids[j], 
                    attention_mask=act_tokens.attention_mask[j],
                    target_input_ids=act_target_tokens.input_ids[j],
                    target_attention_mask=act_target_tokens.attention_mask[j]
                )
            
            tfidf = TfidfVectorizer()
            tfidf.fit(states + observations)

            # For each action find the most relevant previous observation
            obs_embed = tfidf.transform(observations)
            act_embed = tfidf.transform(actions)

            scores = linear_kernel(act_embed, obs_embed)
            mask = np.ones([len(actions), len(observations)])

            for a in range(len(actions)):
                for o in range(len(observations)):
                    if o > a:
                        mask[a, o] = 0

            scores *= mask

            game_idxs += [j for j in range(len(observations))]
            games += [k for _ in observations]
            all_states += states
            all_query_states += query_states
            all_actions += actions
            knowledge_labels += scores.argmax(-1).tolist()
            intent_labels += dialog["act"]

            k += 1

        intent_labels = get_intents(all_actions)

        query_inputs = tokenizer(all_query_states, max_length=args.max_state_length, truncation=True, padding="max_length")
        actions_inputs = tokenizer(all_actions, max_length=args.max_action_length, truncation=True, padding="max_length")
        sample_idxs = [i for i in range(len(all_actions))]

        with tokenizer.as_target_tokenizer(): 
            all_actions = [a + tokenizer.generator.eos_token for a in all_actions] 
            states_inputs = tokenizer(all_states, max_length=args.max_state_length, truncation=True)
            target_actions_inputs = tokenizer(all_actions, max_length=args.max_action_length, truncation=True)

            states_actions = [s + [tokenizer.generator.eos_token_id] + a for s, a in zip(states_inputs.input_ids, target_actions_inputs.input_ids)]  
            mask = [[0] * (len(s) + 1) + [1] * len(a) for s, a in zip(states_inputs.input_ids, target_actions_inputs.input_ids)] 
            labels = []
            for sent, mm in zip(states_actions, mask):
                labels.append([idx if m == 1 else -100 for idx, m in zip(sent, mm)])

        dict_ = {
            "input_ids": states_actions, 
            "query_input_ids": query_inputs.input_ids,
            "query_attention_mask": query_inputs.attention_mask,
            "labels": labels,
            "action_input_ids": actions_inputs.input_ids,
            "action_attention_mask": actions_inputs.attention_mask,
            "sample_idxs": sample_idxs,
            "game_idxs": game_idxs,
            "games": games,
            "knowledge_labels": knowledge_labels,
            "intent_labels": intent_labels
        }

        dataset = Dataset.from_dict(dict_)

        dataset_dict[split] = dataset

    return dataset_dict, game_history, action_history

def _process_clubfloyd_generator_controller(files, tokenizer, args) -> Dataset:
    dataset_dict = {}
    random.shuffle(files)
    idx = int(len(files)*args.test_size)
    train_files = files[:-idx]
    test_files = files[-idx:]

    game_history = GameHistory(args.max_history_length)
    action_history = GameHistory(args.max_action_length)
    k = 0                               
    for split, fs in [("train", train_files), ("test", test_files)]:

        all_states = []
        all_query_states = []
        all_actions = []
        games = []
        game_idxs = []
        knowledge_labels = []

        for i, file in enumerate(fs):

            if args.debug:
                if i == 3:
                    break

            with open(os.path.join(args.data_directory, file)) as f:
                lines = f.readlines()
                states, observations, actions = _split_into_states_actions(lines, tokenizer.generator.eos_token, return_observations=True)
                query_states = _split_into_states_actions(lines, tokenizer.question_encoder.sep_token)[0]

                if states == [] or observations == []:
                    continue

                obs_tokens = tokenizer(observations, max_length=args.max_history_length, padding="max_length", truncation=True)
                with tokenizer.as_target_tokenizer():
                    obs_target_tokens = tokenizer(observations, max_length=args.max_history_length, padding="max_length", truncation=True)
                act_tokens = tokenizer(actions, max_length=args.max_action_length, padding="max_length", truncation=True)
                with tokenizer.as_target_tokenizer():
                    act_target_tokens = tokenizer(actions, max_length=args.max_history_length, padding="max_length", truncation=True) 
                for j in range(len(observations)):
                    game_history.push(
                        game=k, 
                        input_ids=obs_tokens.input_ids[j], 
                        attention_mask=obs_tokens.attention_mask[j],
                        target_input_ids=obs_target_tokens.input_ids[j],
                        target_attention_mask=obs_target_tokens.attention_mask[j]
                    )
                    action_history.push(
                        game=k,
                        input_ids=act_tokens.input_ids[j], 
                        attention_mask=act_tokens.attention_mask[j],
                        target_input_ids=act_target_tokens.input_ids[j],
                        target_attention_mask=act_target_tokens.attention_mask[j]
                    )
                
                tfidf = TfidfVectorizer()
                tfidf.fit(states + observations)

                # For each action find the most relevant previous observation
                obs_embed = tfidf.transform(observations)
                act_embed = tfidf.transform(actions)

                scores = linear_kernel(act_embed, obs_embed)
                mask = np.ones([len(actions), len(observations)])

                for a in range(len(actions)):
                    for o in range(len(observations)):
                        if o > a:
                            mask[a, o] = 0

                scores *= mask

                game_idxs += [j for j in range(len(observations))]
                games += [k for _ in observations]
                all_states += states
                all_query_states += query_states
                all_actions += actions
                knowledge_labels += scores.argmax(-1).tolist()

                k += 1

        intent_labels = get_intents(all_actions)

        query_inputs = tokenizer(all_query_states, max_length=args.max_state_length, padding="max_length")
        query_inputs.input_ids = [ids[-args.max_state_length:] for ids in query_inputs.input_ids]
        query_inputs.attention_mask = [ids[-args.max_state_length:] for ids in query_inputs.attention_mask]
        actions_inputs = tokenizer(all_actions, max_length=args.max_action_length, padding="max_length")
        sample_idxs = [i for i in range(len(all_actions))]

        with tokenizer.as_target_tokenizer(): 
            all_actions = [a + tokenizer.generator.eos_token for a in all_actions] 
            states_inputs = tokenizer(all_states, max_length=args.max_state_length)
            states_inputs.input_ids = [ids[-args.max_state_length:] for ids in states_inputs.input_ids]
            target_actions_inputs = tokenizer(all_actions, max_length=args.max_action_length, truncation=True)

            states_actions = [s + [tokenizer.generator.eos_token_id] + a for s, a in zip(states_inputs.input_ids, target_actions_inputs.input_ids)]  
            mask = [[0] * (len(s) + 1) + [1] * len(a) for s, a in zip(states_inputs.input_ids, target_actions_inputs.input_ids)] 
            labels = []
            for sent, mm in zip(states_actions, mask):
                labels.append([idx if m == 1 else -100 for idx, m in zip(sent, mm)])

        dict_ = {
            "input_ids": states_actions, 
            "query_input_ids": query_inputs.input_ids,
            "query_attention_mask": query_inputs.attention_mask,
            "labels": labels,
            "action_input_ids": actions_inputs.input_ids,
            "action_attention_mask": actions_inputs.attention_mask,
            "sample_idxs": sample_idxs,
            "game_idxs": game_idxs,
            "games": games,
            "knowledge_labels": knowledge_labels,
            "intent_labels": intent_labels
        }

        dataset = Dataset.from_dict(dict_)

        dataset_dict[split] = dataset

    return dataset_dict, game_history, action_history


def _get_dataset_clubfloyd(args, tokenizer):
    
    # Jericho games we want to exclude from data
    exclude = ['intfic_clubfloyd_20090402.html', \
                    'intfic_clubfloyd_20090904.html', \
                    'intfic_clubfloyd_20160401.html', \
                    'intfic_clubfloyd_20160401.txt', \
                    'intfic_clubfloyd_20160701.html', \
                    'intfic_clubfloyd_20161102.html', \
                    'intfic_clubfloyd_20170104.html', \
                    'intfic_clubfloyd_20100903.html', \
                    'intfic_clubfloyd_20080601.html', \
                    "intfic_clubfloyd_20140103.html"]  # this one is blank
    files = [s for s in os.listdir(args.data_directory) if s not in exclude]

    if args.model_type == ModelType.retriever:
        dataset_dict = _process_clubfloyd_retriever(files, tokenizer, args)
        dataset_dict = DatasetDict({"train": dataset_dict["train"], "test": dataset_dict["test"]})
        return dataset_dict

    dataset_dict, game_history, action_history = _process_clubfloyd_generator_controller(files, tokenizer, args)

    dataset_dict = DatasetDict({"train": dataset_dict["train"], "test": dataset_dict["test"]})

    return dataset_dict, game_history, action_history


def _process_jericho(args, tokenizer):
    # For test we include each valid action with broadcast observation
    game_history = GameHistory(args.max_history_length)
    action_history = GameHistory(args.max_action_length)

    games = []
    game_idxs = []
    all_actions = []
    all_states = []
    all_query_states = []

    with jsonlines.open(args.data_directory) as reader:
        for line in reader:
            state = line["obs"].split("[SEP]")
            prev_obs, prev_act, obs = state[0], state[1], state[2]
            act = line["valid_act"][0]
            act = act.split()
            act = " ".join([w if w != "x" else "examine" for w in act])

            obs_tokens = tokenizer([obs], max_length=args.max_history_length, padding="max_length", truncation=True)
            with tokenizer.as_target_tokenizer():
                obs_target_tokens = tokenizer([obs], max_length=args.max_history_length, padding="max_length", truncation=True)
            act_tokens = tokenizer([prev_act], max_length=args.max_action_length, padding="max_length", truncation=True)
            with tokenizer.as_target_tokenizer():
                act_target_tokens = tokenizer([prev_act], max_length=args.max_history_length, padding="max_length", truncation=True) 

            game_history.push(
                game=line["game"], 
                input_ids=obs_tokens.input_ids[0], 
                attention_mask=obs_tokens.attention_mask[0],
                target_input_ids=obs_target_tokens.input_ids[0],
                target_attention_mask=obs_target_tokens.attention_mask[0]
            )
            action_history.push(
                game=line["game"],
                input_ids=act_tokens.input_ids[0], 
                attention_mask=act_tokens.attention_mask[0],
                target_input_ids=act_target_tokens.input_ids[0],
                target_attention_mask=act_target_tokens.attention_mask[0]
            )

            game_idx = len(game_history.game2hist[line["game"]])

            if prev_obs.replace(" ", "") == "[PAD]":
                prev_obs = tokenizer.generator.pad_token
            if prev_act.replace(" ", "") == "[PAD]":
                prev_act = tokenizer.generator.pad_token
            
            state = tokenizer.generator.eos_token.join([prev_obs, prev_act, obs]) + tokenizer.generator.eos_token

            all_query_states.append(line["obs"])
            all_states.append(state)
            all_actions.append(act)
            game_idxs.append(game_idx)
            games.append(line["game"])


    query_inputs = tokenizer(all_query_states, max_length=args.max_state_length, padding="max_length")
    query_inputs.input_ids = [ids[-args.max_state_length:] for ids in query_inputs.input_ids]
    query_inputs.attention_mask = [ids[-args.max_state_length:] for ids in query_inputs.attention_mask]
    with tokenizer.as_target_tokenizer():
        states_inputs = tokenizer(all_states, max_length=args.max_state_length, padding="max_length")
        states_inputs.input_ids = [ids[-args.max_state_length:] for ids in states_inputs.input_ids]

    dict_ = {
        "input_ids": states_inputs.input_ids,
        "query_input_ids": query_inputs.input_ids,
        "query_attention_mask": query_inputs.attention_mask,
        "games": games,
        "game_idxs": game_idxs,
        "act": all_actions
    }

    dataset = Dataset.from_dict(dict_)

    return dataset, game_history, action_history


def _get_dataset_jericho(args, tokenizer):

    return _process_jericho(args, tokenizer)


def get_dataset(args):

    model_type = ModelType(args.model_type)
    task_type = TaskType(args.task_type)
    
    if args.dataset_load_path is not None:
        dataset_dict = DatasetDict.load_from_disk(args.dataset_load_path)
        if model_type in [ModelType.generator, ModelType.controller]:
            game_history = pickle.load(open(args.game_history_path, "rb"))
            action_history = pickle.load(open(args.action_history_path, "rb"))
            return dataset_dict, game_history, action_history

        return dataset_dict

    if model_type in [ModelType.generator, ModelType.controller]:
        tokenizer = get_tokenizer(args)
    elif model_type == ModelType.retriever:
        tokenizer = BertTokenizer.from_pretrained(args.model_path)
    else:
        raise NotImplementedError()
        
    if task_type == TaskType.clubfloyd:
        return _get_dataset_clubfloyd(args, tokenizer)
    elif task_type == TaskType.jericho:
        return _get_dataset_jericho(args, tokenizer)
    
    raise ValueError("task_type not recognised!")
