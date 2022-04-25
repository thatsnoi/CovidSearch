from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


def generate_queries(paragraph, model, tokenizer, num_sentences=3):
    input_ids = tokenizer.encode(paragraph, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=64,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=num_sentences)
    queries = []
    for i in range(len(outputs)):
        query = tokenizer.decode(outputs[i], skip_special_tokens=True)
        queries.append(query)

    return queries


if __name__ == "__main__":
    para = "The World Health Organization declared the SARS-CoV-2 outbreak a global public health emergency. We performed genetic analyses of eighty-six complete or near-complete genomes of SARS-CoV-2 and revealed many mutations and deletions on coding and non-coding regions."
    tokenizer = T5Tokenizer.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
    model = T5ForConditionalGeneration.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
    model.eval()
    generate_queries(para, model, tokenizer)
