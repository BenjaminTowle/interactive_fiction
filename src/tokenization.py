from transformers import GPT2Tokenizer, BertTokenizer, RagTokenizer

def get_tokenizer(args):
    if args.model_type in ["retriever_generator", "controller"]:
        generator = GPT2Tokenizer.from_pretrained(args.generator_tokenizer)
        generator.pad_token = generator.bos_token
        generator.eos_token = generator.bos_token
        generator.add_special_tokens({'cls_token': '[CLS]', 'sep_token': '[SEP]'})
        question_encoder = BertTokenizer.from_pretrained(args.retriever_tokenizer)
        tokenizer = RagTokenizer(question_encoder=question_encoder, generator=generator)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.retriever_tokenizer)

    return tokenizer
