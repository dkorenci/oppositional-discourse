This repository contains the data (the 'XAI-DisInfodemics' corpus) and code for the paper:
What Distinguishes Conspiracy from Critical Narratives? A Computational Analysis of Oppositional Discourse
https://onlinelibrary.wiley.com/doi/10.1111/exsy.13671

The code is licensed under the APACHE 2.0 license, see LICENSE file.
The exception is the span_f1_metric.py file, which is licensed under the GPL license, see the file for details.
The data is licensed under the CC BY-SA 4.0 license, see the LICENSE-DATA file for details.

The data comprising the XAI-DisInfodemics corpus can be found in the 'dataset' folder.

The .json files contain annotated English and Spanish corpora, in a structured format.
Each .json file contains all the texts in the corresponding language, coupled with text- and span-level annotations.
The 'all_annotations' files contain the annotations of all the annotators, while
the 'gold' files contain texts annotated with gold (aggregated) annotations.
The process of obtaining the gold labels is described in the paper.

The sample folders contain samples from the English and Spanish dataset, in a readable .txt format.
For each language, the sample contains 25 annotated files (2 span annotators and gold labels).
Each file contains a text on the first line, followed by the annotations.
