'''
Functionality for calculating statistics regarding critical-conspiracy annotations,
span-level annotations, and join statistics between the two.
'''
from data_tools.classif_data.data_loaders import CLASSIF_DF_ID_COLUMN
from data_tools.classif_data.data_utils import CLS_BINARY_IX
from data_tools.raw_dataset_loaders import load_experiment_datasets
from collections import defaultdict

from tabulate import tabulate

from sequence_labeling.lab_experim_v0 import TASK_LABELS


def print_table(stats, LABELS, CLASS_NAMES):
    headers = ["Class"] + LABELS

    rows = []
    for cls, cls_stats in stats.items():
        class_name = CLASS_NAMES.get(cls, cls)  # Fetch the class name or use 'all' if not found
        row = [class_name]

        # Combine count and percentage for each label in each cell
        for label in LABELS:
            count = cls_stats['counts'].get(label, 0)
            percentage = cls_stats['percentages'].get(label, 0.0)
            combined_value = "{} ({:.2f}%)".format(count, percentage)
            row.append(combined_value)

        rows.append(row)

    # Print the table
    table = tabulate(rows, headers=headers, tablefmt='grid')
    print(table)

def print_latex_table_single(stats, LABELS, CLASS_NAMES):
    # Starting the LaTeX table environment
    output = "\\begin{table}[h!]\n"
    output += "\\centering\n"
    output += "\\begin{tabular}{" + "|".join(["l"] + ["c"] * len(LABELS)) + "}\n"
    output += "\\hline\n"

    # Adding the header row
    output += " & ".join(["Class"] + LABELS) + " \\\\ \\hline\n"

    # Adding each row of stats
    for cls, cls_stats in stats.items():
        class_name = CLASS_NAMES.get(cls, cls)
        row = [class_name]

        for label in LABELS:
            count = cls_stats['counts'].get(label, 0)
            percentage = cls_stats['percentages'].get(label, 0.0)
            combined_value = "{} ({:.2f}\\%)".format(count, percentage)  # Use \\% for percentage in LaTeX
            row.append(combined_value)

        output += " & ".join(row) + " \\\\ \\hline\n"

    # Closing the LaTeX table environment
    output += "\\end{tabular}\n"
    output += "\\end{table}\n"

    print(output)


def print_latex_table_stacked(stats, LABELS, CLASS_NAMES, lang_order):
    # Starting the LaTeX table* environment for two-column wide tables
    output = "\\begin{table*}[h!]\n"
    output += "\\centering\n"

    # Determine the total columns: Empty cell + 'Class' + number of LABELS
    num_cols = 1 + len(LABELS)

    output += "\\begin{tabular}{" + "|".join(["l"] + ["l"] + ["c"] * len(LABELS)) + "}\n"
    output += "\\hline\n"

    # Adding the header row
    output += " & " + " & ".join([""] + LABELS) + " \\\\ \\hline\n"

    # Looping over languages as per given order
    for idx, lang in enumerate(lang_order):
        raw_stats = stats[lang]

        # First stats row for the current language, including language name with rowspan
        cls, cls_stats = list(raw_stats.items())[0]
        class_name = CLASS_NAMES.get(cls, cls)
        row = [f"\\multirow{{3}}{{*}}{{{lang}}}", class_name]

        for label in LABELS:
            count = cls_stats['counts'].get(label, 0)
            percentage = cls_stats['percentages'].get(label, 0.0)
            combined_value = "{} ({:.1f}\\%)".format(count, percentage)  # Use \\% for percentage in LaTeX
            row.append(combined_value)

        output += " & ".join(row) + " \\\\ \n"

        # Remaining stats rows for the current language
        for cls, cls_stats in list(raw_stats.items())[1:]:
            class_name = CLASS_NAMES.get(cls, cls)
            row = ["", class_name]

            for label in LABELS:
                count = cls_stats['counts'].get(label, 0)
                percentage = cls_stats['percentages'].get(label, 0.0)
                combined_value = "{} ({:.1f}\\%)".format(count, percentage)
                row.append(combined_value)

            output += " & ".join(row) + " \\\\ \n"

        if idx != len(lang_order) - 1:  # If not the last language, add an \hline
            output += "\\hline\n"

    # Closing the LaTeX table* environment
    output += "\\end{tabular}\n"
    output += "\\end{table*}\n"

    print(output)


def span_categ_per_binary_categ_stats(lang, gold=True, annot_data=False, print=True):
    cdf, spans = load_experiment_datasets(lang, gold=gold, annot_data=annot_data)
    # Getting column index for the constants and mapping id to binary class
    id_col_index = cdf.columns.get_loc(CLASSIF_DF_ID_COLUMN)
    cls_binary_index = cdf.columns.get_loc(CLS_BINARY_IX)
    id_to_bincls = {row[id_col_index]: row[cls_binary_index] for row in cdf.itertuples(index=False)}
    # Initialize label counters for 0, 1 and all classified docs
    label_counts = {
        'all': defaultdict(int),
        0: defaultdict(int),
        1: defaultdict(int)
    }

    # Count labels
    for span in spans:
        bin_cls = id_to_bincls.get(span.text_id, None)
        if bin_cls is not None:
            label_counts[bin_cls][span.label] += 1
        label_counts['all'][span.label] += 1

    # Calculate statistics: count and percentage
    total_0 = sum(label_counts[0].values())
    total_1 = sum(label_counts[1].values())
    total_all = sum(label_counts['all'].values())

    stats = {
        'all': {
            'counts': label_counts['all'],
            'percentages': {k: (v/total_all) * 100 for k, v in label_counts['all'].items()}
        },
        0: {
            'counts': label_counts[0],
            'percentages': {k: (v/total_0) * 100 for k, v in label_counts[0].items()} if total_0 != 0 else {}
        },
        1: {
            'counts': label_counts[1],
            'percentages': {k: (v/total_1) * 100 for k, v in label_counts[1].items()} if total_1 != 0 else {}
        }
    }
    if print:
        print_table(stats, LABELS=TASK_LABELS, CLASS_NAMES={0: 'Conspiracy', 1: 'Critical', 'all': 'All'})
    return stats

def print_all_tables_txt():
    print("English ALL SPANS:")
    span_categ_per_binary_categ_stats('en', gold=False, annot_data=True)
    print("English GOLD:")
    span_categ_per_binary_categ_stats('en', gold=True, annot_data=False)
    print("Spanish ALL SPANS:")
    span_categ_per_binary_categ_stats('es', gold=False, annot_data=True)
    print("Spanish GOLD:")
    span_categ_per_binary_categ_stats('es', gold=True, annot_data=False)

def create_cumul_latex_table(gold=False, annot_data=True):
    stats_en = span_categ_per_binary_categ_stats('en', gold=gold, annot_data=annot_data)
    stats_es = span_categ_per_binary_categ_stats('es', gold=gold, annot_data=annot_data)
    stats = {'en': stats_en, 'es': stats_es}
    lang_order = ['es', 'en']
    labels = ['A', 'F', 'P', 'V', 'O', 'E']
    assert set(labels) == set(TASK_LABELS)
    print_latex_table_stacked(stats=stats, LABELS=labels, CLASS_NAMES={0: 'Consp.', 1: 'Critic.', 'all': 'All'},
                              lang_order=lang_order)

if __name__ == '__main__':
    #span_categ_per_binary_categ_stats('es')
    #span_categ_per_binary_categ_stats('en')
    #print_all_tables_txt()
    create_cumul_latex_table(gold=True, annot_data=False)