# %% load model
import transformers
import torch

model_id = "models/ragarwal__factual-consistency-llama3-8b"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="mps",
)

# %% run pipeline
PROMPT = """Can the target text be inferred from the source text? In other words, is the target text factually consistent with the source text? Answer with a "Yes" or "No"."""

target = "Die Initiative für eine 13. AHV-Rente möchte, dass Rentner in der Schweiz künftig 13 Mal im Jahr ihre AHV-Rente erhalten, was eine Erhöhung der jährlichen Rente um 8,3 Prozent bedeutet. Dies würde vielen Rentnern helfen, besser über die Runden zu kommen, da die Lebenshaltungskosten steigen. Die Initiative sieht vor, dass auch Rentner mit geringem Einkommen von dieser zusätzlichen Rente profitieren können. Das Komitee, das die Initiative unterstützt, glaubt, dass die AHV genug Geld hat, um diese zusätzliche Zahlung zu finanzieren, und schlägt vor, einen kleinen zusätzlichen Beitrag von den Löhnen zu erheben, um die Kosten langfristig zu decken. Der Bundesrat warnt jedoch, dass die Initiative die AHV finanziell belasten könnte und die Finanzierung unklar bleibt, was zu höheren Kosten für die Bevölkerung führen könnte."

source = (
    "Titel: Für ein besseres Leben im Alter (Initiative für eine 13. AHV-Rente)\nIm Detail: Der Auftrag der AHV\nDie Alters- und Hinterlassenenversicherung (AHV) ist das wichtigste Sozialwerk der Schweiz: Alle Menschen in der Schweiz haben im Alter Anspruch auf eine Rente der AHV. Die Verfassung legt fest, dass die AHV-Renten den Existenzbedarf angemessen decken müssen. Die Mehrheit der Pensionierten bestreitet ihren Lebensunterhalt mit zusätzlichen Einkünften, insbesondere mit Renten aus der Pensionskasse. Wer den Existenzbedarf damit nicht decken kann, hat Anspruch auf Ergänzungsleistungen (EL)",
)


# @todo: optimize args
"""
text_inputs (`str`, `List[str]`, List[Dict[str, str]], or `List[List[Dict[str, str]]]`):
    One or several prompts (or one list of prompts) to complete. If strings or a list of string are
    passed, this pipeline will continue each prompt. Alternatively, a "chat", in the form of a list
    of dicts with "role" and "content" keys, can be passed, or a list of such chats. When chats are passed,
    the model's chat template will be used to format them before passing them to the model.
return_tensors (`bool`, *optional*, defaults to `False`):
    Whether or not to return the tensors of predictions (as token indices) in the outputs. If set to
    `True`, the decoded text is not returned.
return_text (`bool`, *optional*, defaults to `True`):
    Whether or not to return the decoded texts in the outputs.
return_full_text (`bool`, *optional*, defaults to `True`):
    If set to `False` only added text is returned, otherwise the full text is returned. Only meaningful if
    *return_text* is set to True.
clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
    Whether or not to clean up the potential extra spaces in the text output.
continue_final_message( `bool`, *optional*): This indicates that you want the model to continue the
    last message in the input chat rather than starting a new one, allowing you to "prefill" its response.
    By default this is `True` when the final message in the input chat has the `assistant` role and
    `False` otherwise, but you can manually override that behaviour by setting this flag.
prefix (`str`, *optional*):
    Prefix added to prompt.
handle_long_generation (`str`, *optional*):
    By default, this pipelines does not handle long generation (ones that exceed in one form or the other
    the model maximum length). There is no perfect way to adress this (more info
    :https://github.com/huggingface/transformers/issues/14033#issuecomment-948385227). This provides common
    strategies to work around that problem depending on your use case.

    - `None` : default strategy where nothing in particular happens
    - `"hole"`: Truncates left of input, and leaves a gap wide enough to let generation happen (might
      truncate a lot of the prompt and not suitable when generation exceed the model capacity)
generate_kwargs (`dict`, *optional*):
    Additional keyword arguments to pass along to the generate method of the model (see the generate method
    corresponding to your framework [here](./text_generation)).
"""
answer = pipeline(
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
    generated_text = item.get("generated_text", [])
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
