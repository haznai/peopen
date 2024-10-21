import transformers
import torch
from pathlib import Path
import os


# %% helper functions
def load_model() -> transformers.TextGenerationPipeline:
    """
    This function works in both script and Jupyter notebook environments.
    It tries different methods to find the safetensors of the factual-consistency model in the `models` directory.

    Returns:
        TextGenerationPipeline: The transformers pipeline object for the factual-consistency model.
    """
    try:
        # Try to get the path of the current file (works in scripts)
        path = Path(__file__).resolve().parent.parent.parent
    except NameError:
        try:
            # Try to get the path from Jupyter notebook
            import IPython

            path = Path(IPython.get_ipython().kernel.profile_dir).parent.parent.parent  # type: ignore
        except Exception:
            # Fallback to current working directory
            path = Path(os.getcwd()).parent.parent.resolve()

    model_path = path.joinpath("models", "ragarwal__factual-consistency-llama3-8b")

    pipeline = transformers.pipeline(
        "text-generation",
        model=str(model_path),
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="mps",
    )

    return pipeline  # type: ignore


# %% model loading
LMFactualConsistency = load_model()

# %% text generation
PROMPT = """Can the target text be inferred from the source text? In other words, is the target text factually consistent with the source text? Answer with a "Yes" or "No"."""

target = "Die Initiative für eine 13. AHV-Rente möchte, dass Rentner in der Schweiz künftig 13 Mal im Jahr ihre AHV-Rente erhalten, was eine Erhöhung der jährlichen Rente um 8,3 Prozent bedeutet. Dies würde vielen Rentnern helfen, besser über die Runden zu kommen, da die Lebenshaltungskosten steigen. Die Initiative sieht vor, dass auch Rentner mit geringem Einkommen von dieser zusätzlichen Rente profitieren können. Das Komitee, das die Initiative unterstützt, glaubt, dass die AHV genug Geld hat, um diese zusätzliche Zahlung zu finanzieren, und schlägt vor, einen kleinen zusätzlichen Beitrag von den Löhnen zu erheben, um die Kosten langfristig zu decken. Der Bundesrat warnt jedoch, dass die Initiative die AHV finanziell belasten könnte und die Finanzierung unklar bleibt, was zu höheren Kosten für die Bevölkerung führen könnte."

source = (
    "Titel: Für ein besseres Leben im Alter (Initiative für eine 13. AHV-Rente)\nIm Detail: Der Auftrag der AHV\nDie Alters- und Hinterlassenenversicherung (AHV) ist das wichtigste Sozialwerk der Schweiz: Alle Menschen in der Schweiz haben im Alter Anspruch auf eine Rente der AHV. Die Verfassung legt fest, dass die AHV-Renten den Existenzbedarf angemessen decken müssen. Die Mehrheit der Pensionierten bestreitet ihren Lebensunterhalt mit zusätzlichen Einkünften, insbesondere mit Renten aus der Pensionskasse. Wer den Existenzbedarf damit nicht decken kann, hat Anspruch auf Ergänzungsleistungen (EL)",
)

answer = LMFactualConsistency(
    [
        {
            "role": "user",
            "content": f"{PROMPT}\n\n_target_ {target}\n\n_source_ {source}",
        }
    ],
    max_length=4096,
    clean_up_tokenization_spaces=True,
)

# Initialize a variable to hold the assistant's content
assistant_content = None

# Navigate through the structure to find the assistant's message
for item in answer:  # type: ignore
    generated_text = item.get("generated_text", [])  # type: ignore
    for message in generated_text:
        if message.get("role") == "assistant":
            assistant_content = message.get("content")
            break  # Exit the inner loop once found
    if assistant_content:
        break  # Exit the outer loop if assistant's content is found

# Output the assistant's content
if assistant_content is None:
    raise ValueError("Assistant's content not found in the pipeline output.")
print(assistant_content)
