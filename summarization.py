import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import pandas as pd
from typing import List, Dict, Tuple, Any

INPUT_CSV = "summarization_input.csv"
OUTPUT_CSV = "summarization_output.csv"
MIN_LENGTH = 30
MAX_LENGTH = 100

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
device = torch.device("cpu")


def summarize(text: str, min_length: int = 30, max_length: int = 100):
    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_Text = "summarize: " + preprocess_text
    print("original text preprocessed: \n", preprocess_text)

    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

    # summmarize
    summary_ids = model.generate(
        tokenized_text,
        num_beams=4,
        no_repeat_ngram_size=2,
        min_length=30,
        max_length=100,
        early_stopping=True,
    )

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    capitalized_output = ". ".join(
        map(lambda s: s.strip().capitalize(), output.split("."))
    ).strip()

    return capitalized_output


def read_input_csv(csv_filename: str) -> pd.DataFrame:
    df = pd.read_csv(csv_filename)
    return df


def get_texts_from_csv(csv_filename: str) -> List[str]:
    df = read_input_csv(csv_filename)
    text_list = df["text"].tolist()
    return text_list


def process_summarization(input_csv: str, output_csv: str):
    text_list = get_texts_from_csv(input_csv)
    processed_data = [
        {
            "original": text,
            "summary": summarize(text, min_length=MIN_LENGTH, max_length=MAX_LENGTH),
        }
        for text in text_list
    ]
    df = pd.DataFrame(processed_data)
    print(df)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    process_summarization(INPUT_CSV, OUTPUT_CSV)
