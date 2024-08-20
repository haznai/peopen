# meant to be run after structured_data_from_broshures.py
# structure after runnign structured_data_from_broshures.py:
#
# data/processed/brochure_ocr/
#    - YYYY_MM_DD/ <- folder for every brochure
#       - YYYY_MM_DD.md <- ocr content
#       - YYYY_MM_DD_meta.json <- meta data
#       - X_image_X.png <- image
#
# the end goal is to have a folder for every volksinitiative
# data/processed/volksinitiatives/
#     - YYYY_MM_DD/ <- folder from which broshure the volksiniative came
#        - volksinitiative_1.md <- ocr content of the volksinitiative
#        - volksinitiative_2.md
#        - volksinitiative_3.md
#        - ...

import os
import re

# ----- Load in the broshure content -----
broshure_md_files = { date: f"data/processed/brochure_ocr/{date}/{date}.md" for date in os.listdir("data/processed/brochure_ocr/")}
# If .DS_Store is in the dict, remove it
if ".DS_Store" in broshure_md_files:
    del broshure_md_files[".DS_Store"]
# load in ocr-content from a file
broshure_content = { date: open(file_path, "r").read() for date, file_path in broshure_md_files.items()}

# ----- Split the broshure content by volksinitiative -----
# Search for the patter "im detail" (case insensitive) and split the content by it
#
# ------------------------------------------------
# | Im Detail: Muster für eine Volksinitiative 1 | <- some volksinitiativen just have a single "im detail" match
# |  ...... content of volksinitiative 1 ....... |
# | Im Detail: Muster für eine Volksinitiative 2 | <- split here
# |  ...... content of volksinitiative 2 ....... |
# ------------------------------------------------

volksinitiative_content = {}
for date, content in broshure_content.items():
    split_content = re.split("im detail", content, flags=re.IGNORECASE)
    volksinitiative_content[date] = [volksiniative.strip() for volksiniative in split_content if volksiniative.strip() != ""]

# print the number of volksinitiatives found for each broshure
# sort by date
for date, volksinitiatives in sorted(volksinitiative_content.items()):
    print(f"{date}: {len(volksinitiatives)} volksinitiatives found")

# ----- Save the volksinitiative content -----
for date, volksinitiatives in volksinitiative_content.items():
    # Create a directory for each date
    os.makedirs(f"data/processed/volksinitiatives/{date}", exist_ok=True)

    # Iterate over each volksinitiative
    for i, volksinitiative in enumerate(volksinitiatives, start=1):
        # Create a file for each volksinitiative
        with open(f"data/processed/volksinitiatives/{date}/volksinitiative_{i}.md", "w") as file:
            file.write(volksinitiative)
