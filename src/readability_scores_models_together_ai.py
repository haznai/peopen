# %% imports
import pandas as pd
import textstat

from model_definition import (
    PenPrompterNetwork,
    Dataset,
    LanguageModel,
    get_model_path,
    get_train_and_valid_path,
)

from submodels.factual_consistency_model import (
    FactualConsistencyNetwork,
)

import dspy


# %% load and run
# together.ai model names
language_model_names = [
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
]


# load in the csv file
df = pd.read_csv("../evaluations/model_scores.csv")
df.tail()

# Create new columns for readability scores if they don't exist
if "flesch_reading_ease" not in df.columns:
    df["flesch_reading_ease"] = None
if "flesch_kincaid_grade" not in df.columns:
    df["flesch_kincaid_grade"] = None

# iterate 'model_name' column in df
for index, row in df.iterrows():
    network_model_name = row["model_name"]
    # find the language model that corresponds to the network model
    lm_for_this_network = None
    for language_model_name in language_model_names:
        if language_model_name in network_model_name:
            lm_for_this_network = language_model_name
            print(f"Found matching language model: {lm_for_this_network}")
            break

    if lm_for_this_network is None:
        print(f"No matching language model found for {network_model_name}")
        continue

    # setup the model
    model_path_pp = get_model_path(
        f"penprompter/2024-11-01_{network_model_name.replace("/", "_")}.json"
    )
    # load in best fc mode
    model_path_fc = get_model_path(
        "factual_consistency/2024-11-01_bootsrapfewshotwithrandomsearch_second.json"
    )

    fc_model = FactualConsistencyNetwork()
    fc_model.load(model_path_fc)
    fc_model._compiled = True
    network = PenPrompterNetwork(fc_model)
    trained_network = PenPrompterNetwork(fc_model)
    trained_network.load(model_path_pp)
    lm = LanguageModel(dspy.Together(lm_for_this_network))

    train_path, valid_path = get_train_and_valid_path()
    data = Dataset(train_pickle_path=str(train_path), valid_pickle_path=str(valid_path))
    valset = data.data["validation"]

    # iterate over the validation set and get all preds
    preds = []
    for example in valset:
        pred_text = trained_network.forward(
            titel=example.titel,
            im_detail=example.im_detail,
            argumenteKomitee=example.argumenteKomitee,
            empfehlungKomitee=example.empfehlungKomitee,
            argumenteBundesrat=example.argumenteBundesrat,
            empfehlungBundesrat=example.empfehlungBundesrat,
        ).final_wortlaut

        preds.append(pred_text)

    # calculate readability scores
    textstat.set_lang("de")

    reading_ease_score = 0
    reading_grade_score = 0
    for pred in preds:
        reading_ease_score = reading_ease_score + textstat.flesch_reading_ease(pred)
        reading_grade_score = reading_grade_score + textstat.flesch_kincaid_grade(pred)

    reading_ease_score = reading_ease_score / len(preds)
    reading_grade_score = reading_grade_score / len(preds)
    print(f"{network_model_name} fleisch_reading_ease: {reading_ease_score}")
    print(f"{network_model_name} flesch_kincaid_grade: {reading_grade_score}")

    # Update the DataFrame with the new scores
    df.at[index, "flesch_reading_ease"] = reading_ease_score
    df.at[index, "flesch_kincaid_grade"] = reading_grade_score

# Save the updated DataFrame to a new CSV file
df.to_csv("../evaluations/model_scores_with_readability.csv", index=False)
