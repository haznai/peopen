import marimo

__generated_with = "0.4.7"
app = marimo.App(width="medium")


@app.cell
def __():
    # dspy imports
    import dspy
    from dspy.datasets import DataLoader


    # metric evaluation imports
    from nltk.translate import meteor_score
    import numpy as np
    import tiktoken
    from nltk.translate.bleu_score import sentence_bleu
    from rouge import Rouge
    from evaluate import load


    # typing imports
    from typing import Literal, Mapping
    return (
        DataLoader,
        Literal,
        Mapping,
        Rouge,
        dspy,
        load,
        meteor_score,
        np,
        sentence_bleu,
        tiktoken,
    )


@app.cell
def __(DataLoader, Literal, Mapping, dspy):
    ##### misc functions #####
    def load_lds_dataset() -> (
        Mapping[Literal["train", "validation", "test"], list[dspy.Example]]
    ):
        # only load the text and regeste fields
        # todo: load and evaluate the full dataset
        fields = ("text", "regeste")
        # define input (x in ml) for the model
        input_keys = ("text",)
        dataset = DataLoader().from_huggingface(
            "rcds/swiss_leading_decision_summarization", fields=fields, input_keys=input_keys
        )
        return dataset


    def set_dspy_model(
        open_ai_model_name: Literal["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    ) -> None:
        # get "OPENAI_API_KEY" from '.env' file in pure python
        # it's saved as "OPENAI_API_KEY=YOUR_API_KEY" in the '.env' file
        with open(".env", "r") as f:
            for line in f:
                if "OPENAI_API_KEY" in line:
                    api_key = line.split("=")[1]
                    # strip \n
                    api_key = api_key.strip()
                    break
        model = dspy.OpenAI(model=open_ai_model_name, api_key=api_key)
        dspy.settings.configure(lm=model)
    return load_lds_dataset, set_dspy_model


@app.cell
def __(load_lds_dataset):
    test_dataset = load_lds_dataset()
    print("inputs: ", test_dataset["train"][0].inputs())
    print("outputs: ", test_dataset["train"][0].labels())
    return test_dataset,


@app.cell
def __(dspy):
    #### dspy classes ####
    class DecisionSummarizationSignature(dspy.Signature):
        """ "Ziel: Generiere eine Regeste basierend auf einem Schweizer Gerichtsurteils.\nHintergrund: Ein Schweizer Gerichtsurteil setzt sich aus Sachverhalt, Erwägungen und Dispositiv zusammen. Die Regeste dient als Kurzzusammenfassung und beinhaltet Leitsätze des Urteils. Nur Leitentscheide haben eine Regeste.\nAnweisung:\n1. Sachverhalt: Lies und verstehe den gegebenen Sachverhalt.\n2. Erwägungen: Analysiere die Erwägungen, um die Hauptargumente und Gründe zu identifizieren.\n3. Dispositiv: Beachte das Dispositiv, da es das endgültige Urteil enthält.\n4. Erstelle die Regeste: Die Regeste sollte aus drei sehr kurzen Teilen bestehen: a. Zitiere die wichtigsten relevanten Artikelziffern (ohne den Artikeltitel). b. Nenne kurze, relevante, deskriptive Keywords, über die Thematik des Falls. c. Formuliere einen sehr kurzen Fliesstext, der die wichtigsten Erwägungen zitiert und kurz zusammenfasst.\nOutput: Die Regeste sollte eine klare und strukturierte Kurzzusammenfassung des Urteils bieten, die aus zitierten Artikeln, Keywords und einem sehr kurzen Fliesstext besteht."""

        text = dspy.InputField()
        regeste = dspy.OutputField()
    return DecisionSummarizationSignature,


@app.cell
def __(
    Rouge,
    load,
    meteor_score,
    np,
    open_ai_model_name,
    sentence_bleu,
    tiktoken,
):
    #### metric evaluation functions ####
    def average_rouge_scores(rouge_scores_list):
        avg_scores = {
            "rouge-1": {"r": 0, "p": 0, "f": 0},
            "rouge-2": {"r": 0, "p": 0, "f": 0},
            "rouge-l": {"r": 0, "p": 0, "f": 0},
        }

        num_scores = len(rouge_scores_list)

        for scores in rouge_scores_list:
            for rouge_type in avg_scores:
                for metric in avg_scores[rouge_type]:
                    avg_scores[rouge_type][metric] += scores[rouge_type][metric]

        for rouge_type in avg_scores:
            for metric in avg_scores[rouge_type]:
                avg_scores[rouge_type][metric] /= num_scores

        return avg_scores


    def average_bert_score(bert_scores):
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        count = len(bert_scores)

        for bert_score in bert_scores:
            total_precision += sum(bert_score["precision"]) / len(bert_score["precision"])
            total_recall += sum(bert_score["recall"]) / len(bert_score["recall"])
            total_f1 += sum(bert_score["f1"]) / len(bert_score["f1"])

        return {
            "precision": total_precision / count,
            "recall": total_recall / count,
            "f1": total_f1 / count,
        }


    def tokenize(text, tokenizer):
        enc = get_tokenizer(tokenizer)
        text_token_ids = enc.encode(text)
        # we want a list text_tokens with applying enc.decode([x]) for each x in text_token_ids
        text_tokens = [enc.decode([x]) for x in text_token_ids]
        return text_tokens  ## list of tokens


    def get_tokenizer(tokenizer):
        enc = tiktoken.get_encoding("cl100k_base")
        if "gpt" not in tokenizer[:4]:
            print(f"no score tokenizer for {tokenizer}")
            tokenizer = "gpt-4-turbo"
        print("Loading tokenizer", tokenizer)
        enc = tiktoken.encoding_for_model(tokenizer)
        return enc


    def compute_scores(completion_dataset, num_examples=10):
        scores = {"meteor": [], "rouge": [], "bleu": [], "bert": []}
        rouge = Rouge()
        bertscore = load("bertscore")

        for idx, entry in enumerate(completion_dataset):
            """
            target_text_tokens: list of tokens
            predicted_text_tokens: list of tokens

            predicted_text: string (plain text)
            target_text: string (plain text)

            tokenized_target_text: string (tokens)
            tokenized_predicted_text: string (tokens)
            """
            target_text_tokens = tokenize(entry["target"], open_ai_model_name)
            predicted_text_tokens = tokenize(entry["predicted"], open_ai_model_name)

            predicted_text = entry["predicted"]
            target_text = entry["target"]

            tokenized_target_text = " ".join(target_text_tokens)
            tokenized_predicted_text = " ".join(predicted_text_tokens)

            # Calculate Meteor scores
            meteor = meteor_score.meteor_score([target_text_tokens], predicted_text_tokens)
            scores["meteor"].append(meteor)

            # Calculate Rouge scores
            rouge_scores = rouge.get_scores(predicted_text, target_text)[0]
            scores["rouge"].append(rouge_scores)

            # Calculate Bleu scores
            bleu = sentence_bleu(
                [tokenized_target_text],
                tokenized_predicted_text,
                weights=(0.25, 0.25, 0.25, 0.25),
            )
            scores["bleu"].append(bleu)

            # Calculate BERTScore
            bert = bertscore.compute(
                predictions=[predicted_text],
                references=[target_text],
                model_type="bert-base-multilingual-cased",
                lang=["de", "fr", "it"],
            )
            scores["bert"].append(bert)

            output_examples = []
            output_examples.append(
                {
                    "language": entry["lang"],
                    "input": entry["input"],
                    "target": target_text,
                    "predicted": predicted_text,
                    "meteor": meteor,
                    "bert-f1": bert["f1"][0],
                    "bleu": bleu,
                    "rouge-1_f1": rouge_scores["rouge-1"]["f"],
                    "rouge-2_f1": rouge_scores["rouge-2"]["f"],
                    "rouge-l_f1": rouge_scores["rouge-l"]["f"],
                    "bert_full": bert,
                    "rouge_full": rouge_scores,
                }
            )

            # Print examples
            if idx < num_examples:
                print("\n", flush=True)
                print("#" * 180, flush=True)
                print(f"Example {idx + 1} of {len(completion_dataset)}")
                print(f"Output: {predicted_text}")
                print("-" * 100)
                print(f"Label: {target_text}")
                print("-" * 100)
                print(f"METEOR Score: {meteor:.4f}")
                print(f"ROUGE Score: {rouge_scores}")
                print(f"BLEU Score: {bleu:.4f}")
                print(f"BERTScore: {bert}")
                print("#" * 180, flush=True)
                print("\n", flush=True)

        return (
            (
                np.mean(scores["meteor"]),
                average_rouge_scores(scores["rouge"]),
                np.mean(scores["bleu"]),
                average_bert_score(scores["bert"]),
            ),
            output_examples,
        )
    return (
        average_bert_score,
        average_rouge_scores,
        compute_scores,
        get_tokenizer,
        tokenize,
    )


if __name__ == "__main__":
    app.run()
