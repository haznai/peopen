import pickle
from pydantic import BaseModel


# %% Prepare data
class ArticleNumber(BaseModel):
    number: str


with open("../../data/processed/truncated_wortlaut_train.pickle", "rb") as f:
    train_data = pickle.load(f)

#  %% post-process the data

# first: turn the `ArticleNumbers` into a list of strings
# then: remove everything in the string that isn't a number digit

for i in range(len(train_data)):
    # Convert ArticleNumbers to a list of strings
    article_numbers = [an.number for an in train_data[i]["article_numbers"]]

    # Remove non-digit characters from each article number
    article_numbers = ["".join(filter(str.isdigit, an)) for an in article_numbers]

    # Update the train_data entry with the processed article numbers
    train_data[i]["article_numbers"] = article_numbers


# %% Sanity checks
def run_sanity_checks(data):
    for item in data:
        # Check 1: Ensure 'article_numbers' is a list
        assert isinstance(
            item["article_numbers"], list
        ), f"Article numbers should be a list, got {type(item['article_numbers'])}"

        # Check 2: Ensure all article numbers are strings of digits
        assert all(
            an.isdigit() for an in item["article_numbers"]
        ), f"All article numbers should be strings of digits, got {item['article_numbers']}"

        # Check 3: Ensure 'wortlaut_original' and 'wortlaut_truncated' are non-empty strings
        assert (
            isinstance(item["wortlaut_original"], str)
            and len(item["wortlaut_original"]) > 0
        ), "wortlaut_original should be a non-empty string"
        assert (
            isinstance(item["wortlaut_truncated"], str)
            and len(item["wortlaut_truncated"]) > 0
        ), "wortlaut_truncated should be a non-empty string"

        # Check 4: Ensure there's at least one article number
        assert (
            len(item["article_numbers"]) > 0
        ), "There should be at least one article number"
    print("All sanity checks passed!")


# Run sanity checks
run_sanity_checks(train_data)

# %% Save the processed data (overwrite original file)
with open("../../data/processed/truncated_wortlaut_train.pickle", "wb") as f:
    pickle.dump(train_data, f)

print("Processed data saved successfully!")
