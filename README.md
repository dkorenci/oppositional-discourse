This repository contains the data and code for the paper:
What Distinguishes Conspiracy from Critical Narratives? A Computational Analysis of Oppositional Discourse
https://onlinelibrary.wiley.com/doi/10.1111/exsy.13671

The code is licensed under the APACHE 2.0 license, see LICENSE file.
The exception is the span_f1_metric.py file, which is licensed under the GPL license, see the file for details.
The data is licensed under the CC BY-SA 4.0 license, see the LICENSE-DATA file for details.


The data can be found in the 'dataset' folder

The sample folders contain samples from the English and Spanish dataset, in a readable .txt format.
For each language, the sample contains 25 annotated files (2 span annotators and gold labels).
Each file contains a text on the first line, followed by the annotations.

The .json files contain full datasets, with annotations, in a structured format.
The 'gold' datasets contain gold labels, both for the binary per-text and for the span annotations.
The process of obtaining the gold labels is described in the paper.
The 'all_annotations' datasets contain the annotations of all the annotators, both on text- and span-level.
