from transformers import GPT2Tokenizer, BertTokenizer, RagTokenizer
from src.models import ModelType

def get_tokenizer(args):
    model_type = ModelType(args.model_type)
    if model_type in [ModelType.generator, ModelType.controller]:
        generator = GPT2Tokenizer.from_pretrained(args.generator_tokenizer)
        generator.pad_token = generator.bos_token
        generator.eos_token = generator.bos_token
        generator.add_special_tokens({'cls_token': '[CLS]', 'sep_token': '[SEP]'})
        question_encoder = BertTokenizer.from_pretrained(args.retriever_tokenizer)
        tokenizer = RagTokenizer(question_encoder=question_encoder, generator=generator)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.retriever_tokenizer)

    return tokenizer
