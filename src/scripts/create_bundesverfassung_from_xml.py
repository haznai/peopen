import xml.etree.ElementTree as ET

# Define the namespace used in the XML
namespaces = {"akn": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"}


def extract_text(element, exclude_tags=None):
    """Extract text from an element, excluding specified tags."""
    if exclude_tags is None:
        exclude_tags = []
    text_parts = []
    if element.tag not in exclude_tags and element.text:
        text_parts.append(element.text)
    for child in element:
        text_parts.append(extract_text(child, exclude_tags))
        if child.tail:
            text_parts.append(child.tail)
    return "".join(text_parts)


tree = ET.parse("../../data/raw/SR-101-03032024-DE.xml")
root = tree.getroot()
output = []
footnote_counter = 1

act = root.find(".//{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}act")

# Process preface
preface = act.find("akn:preface", namespaces)
if preface is not None:
    for p in preface.findall("akn:p", namespaces):
        text = extract_text(p)
        output.append(text.strip() + "\n")

# Process preamble
preamble = act.find("akn:preamble", namespaces)
if preamble is not None:
    for p in preamble.findall("akn:p", namespaces):
        role = p.get("{http://fedlex.admin.ch/}role")
        if role == "heading":
            heading_text = extract_text(p).strip()
            output.append(f"##### **{heading_text}**\n")
        else:
            text = extract_text(p)
            output.append(text.strip() + "\n")

# Process body
body = act.find("akn:body", namespaces)
if body is not None:
    for elem in body.iter():
        # Extract local tag name without namespace
        tag = elem.tag
        if "}" in tag:
            tag = tag.split("}", 1)[1]
        else:
            tag = tag
        if tag == "title":
            num = elem.find("akn:num", namespaces)
            heading = elem.find("akn:heading", namespaces)
            if num is not None and heading is not None:
                title_text = f"{num.text.strip()} {heading.text.strip()}"
                underline = "=" * len(title_text)
                output.append(f"{title_text}\n{underline}\n")
        elif tag == "chapter":
            num = elem.find("akn:num", namespaces)
            heading = elem.find("akn:heading", namespaces)
            if num is not None and heading is not None:
                chapter_text = f"{num.text.strip()} {heading.text.strip()}"
                underline = "-" * len(chapter_text)
                output.append(f"{chapter_text}\n{underline}\n")
        elif tag == "article":
            num = elem.find("akn:num", namespaces)
            heading = elem.find("akn:heading", namespaces)
            if num is not None:
                num_text = extract_text(num).strip()
                if heading is not None:
                    heading_text = extract_text(heading).strip()
                    output.append(f"###### **{num_text}** {heading_text}\n")
                else:
                    output.append(f"###### **{num_text}**\n")
            # Process paragraphs
            for para in elem.findall("akn:paragraph", namespaces):
                para_num = para.find("akn:num", namespaces)
                content = para.find("akn:content", namespaces)
                text = ""
                if para_num is not None:
                    text += para_num.text.strip() + " "
                if content is not None:
                    para_text = extract_text(content).strip()
                    text += para_text
                output.append(text.strip() + "\n")
            # Process authorial notes
            for note in elem.findall(".//akn:authorialNote", namespaces):
                note_text = extract_text(
                    note,
                    exclude_tags=[
                        "{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}ref"
                    ],
                ).strip()
                output.append(f"{footnote_counter} {note_text}\n")
                footnote_counter += 1

# Print the output
output = "\n".join(output)
print(output)

# Save the output to a file
with open("../../data/processed/bundesverfassung_de.md", "w") as f:
    f.write(output)
