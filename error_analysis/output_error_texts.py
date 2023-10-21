import shutil

import pandas as pd
import os

def sample_errors(path_to_reftab, path_to_clstab, class_column, sample_size=50, rseed=8954):
    if rseed is not None:
        pd.np.random.seed(rseed)
    # Load the tables
    reftab = pd.read_excel(path_to_reftab)
    clstab = pd.read_excel(path_to_clstab)
    # check that the values in 'text_id' columns are the same, disregarding order, warn otherwise
    if not set(reftab['text_id']) == set(clstab['text_id']):
        print('Warning: text IDs in the two tables are not the same.')
    # Merge the two tables on text_id
    merged = reftab.merge(clstab, on='text_id', suffixes=('_ref', '_cls'))
    # Identify false positives and false negatives
    false_positives = merged[(merged[class_column + '_ref'] == 0) & (merged[class_column + '_cls'] == 1)]
    false_negatives = merged[(merged[class_column + '_ref'] == 1) & (merged[class_column + '_cls'] == 0)]
    # Sample sample_size from each
    false_positives_sample = false_positives.sample(min(sample_size, len(false_positives)))
    false_negatives_sample = false_negatives.sample(min(sample_size, len(false_negatives)))
    # Write the text IDs to .txt files
    with open('false_positives.txt', 'w') as f:
        for text_id in false_positives_sample['text_id']:
            f.write(str(text_id) + '\n')
    with open('false_negatives.txt', 'w') as f:
        for text_id in false_negatives_sample['text_id']:
            f.write(str(text_id) + '\n')
    # Save texts in corresponding folders
    if os.path.exists('false_positives_texts'):
        shutil.rmtree('false_positives_texts')
    os.mkdir('false_positives_texts')
    for text_id, text in zip(false_positives_sample['text_id'], false_positives_sample['text']):
        with open(f'false_positives_texts/text_{text_id}.txt', 'w', encoding="utf-8") as f:
            f.write(text)
    #
    if os.path.exists('false_negatives_texts'):
        shutil.rmtree('false_negatives_texts')
    os.mkdir('false_negatives_texts')
    for text_id, text in zip(false_negatives_sample['text_id'], false_negatives_sample['text']):
        with open(f'false_negatives_texts/text_{text_id}.txt', 'w', encoding="utf-8") as f:
            f.write(text)

if __name__ == '__main__':
    sample_errors(
                  path_to_reftab='es.violence.dataset.xlsx',
                  path_to_clstab='es.span.categ.predict.xlsx',
                  class_column='F'
                  )