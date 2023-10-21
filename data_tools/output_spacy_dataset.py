import os
from pydoc import Doc
from typing import List

from data_tools.create_spacy_dataset import load_spacy_dataset_docbin
from data_tools.seqlab_data.create_spacy_span_dataset import get_doc_id, get_doc_class, get_annoation_tuples_from_doc


def create_text_files_from_docs(docs: List[Doc], labels: List[str], output_dir: str = "./"):
    """
    Create text files from SpaCy Doc objects.
    Each file will have the document text, binary class, and author spans sorted by given labels.

    :param docs: List of SpaCy Doc objects.
    :param labels: List of labels setting the order of span outputs.
    :param output_dir: Directory to save the generated text files.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for doc in docs:
        # Create a new file for each doc
        filename = f"text_{get_doc_id(doc)}.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as file:
            # Write the text of the document
            file.write(doc.text + "\n\n")

            # Write the binary class
            cls = get_doc_class(doc)
            cls_label = "Critical" if cls == 1 else "Conspiracy"
            file.write(f"Binary Class: {cls_label}\n\n")

            # Group annotations by author
            annotations_by_author = {}
            for annot in get_annoation_tuples_from_doc(doc):
                label, start, end, author = annot
                if author not in annotations_by_author:
                    annotations_by_author[author] = []
                annotations_by_author[author].append(annot)

            # Write author spans
            for author, annotations in annotations_by_author.items():
                file.write(f"Author: {author}\n")

                # Sort annotations by given labels
                sorted_annots = sorted(annotations,
                                       key=lambda x: labels.index(x[0]) if x[0] in labels else float('inf'))

                for annot in sorted_annots:
                    label, start, end, _ = annot
                    span_text = doc[start:end].text
                    file.write(f"{label}: {span_text} [{start}, {end}]\n")
                file.write("\n")  # Separate different authors with a newline


def output_spacy_dset(lang):
    docs = load_spacy_dataset_docbin(lang)
    label_order = ['F', 'A', 'P', 'V', 'E', 'O']
    create_text_files_from_docs(docs, label_order, output_dir=f'./{lang}_dset_gold')

if __name__ == '__main__':
    output_spacy_dset('en')
    output_spacy_dset('es')
