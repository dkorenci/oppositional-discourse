import numpy as np
import pyannote.core.segment
from pyannote.core import Segment
from pygamma_agreement import StatisticalContinuumSampler, Continuum

class StatSamplerPerAnnotator(StatisticalContinuumSampler):
    '''
    Calculate span-generating statistic per each annotator.
    Input continuum has annotators names formated like NAME_FILENUM in order to distinguish different files.
    Per-annotator statistics are calculated for 'base' annotator names, ie NAME, and
    continuums are generated with these names.
    '''

    @staticmethod
    def _annotator(nannotator):
        ''' Extracts the annotator name form 'numbered' annotator name of the form name_filenum '''
        return nannotator.split('_')[0]

    def _annotators(self):
        ''' The set of annotator names without the filenum numbers. '''
        annots = set(self._annotator(nann) for nann in self._reference_continuum.annotators)
        return list(annots)

    def _set_gap_information(self):
        # To prevent glitching continua with 1 unit
        gaps = { ann: [0] for ann in self._annotators() }
        current_annotator = None
        last_unit = None
        for nannotator, unit in self._reference_continuum:
            annotator = self._annotator(nannotator)
            if current_annotator != nannotator:
                current_annotator = nannotator
            else:
                gaps[annotator].append(unit.segment.start - last_unit.segment.end)
            last_unit = unit
        for nannotator, annotation_set in self._reference_continuum._annotations.items():
            if len(annotation_set) == 0: continue
            annotator = self._annotator(nannotator)
            if annotation_set[0].segment.start > 0:
                gaps[annotator].append(annotation_set[0].segment.start)
        self._avg_gap, self._std_gap = {}, {}
        for ann in self._annotators():
            self._avg_gap[ann] = float(np.mean(gaps[ann]))
            self._std_gap[ann] = float(np.std(gaps[ann]))

    def _set_duration_information(self):
        durations = { ann: [] for ann in self._annotators() }
        for nannotator, unit in self._reference_continuum:
            annotator = self._annotator(nannotator)
            durations[annotator].append(unit.segment.duration)
        self._avg_unit_duration, self._std_unit_duration = {}, {}
        for ann in self._annotators():
            self._avg_unit_duration[ann] = float(np.mean(durations[ann]))
            self._std_unit_duration[ann] = float(np.std(durations[ann]))

    def _set_categories_information(self):
        # init data
        categories_set = self._reference_continuum.categories
        self._categories = np.array(categories_set)
        self._categories_weight = {}
        for ann in self._annotators():
            self._categories_weight[ann] = np.zeros(len(categories_set))
        # count category occurences, per annotator
        for nannotator, unit in self._reference_continuum:
            ann = self._annotator(nannotator)
            self._categories_weight[ann][categories_set.index(unit.annotation)] += 1
        # normalize weights to probability distributions
        for ann in self._annotators():
            self._categories_weight[ann] /= sum(self._categories_weight[ann])

    def _set_nb_units_information(self):
        lens_per_annotator = {}
        for nannotator, annotations in self._reference_continuum._annotations.items():
            #print(nannotator, len(annotations))
            annotator = self._annotator(nannotator)
            if annotator in lens_per_annotator: lens_per_annotator[annotator].append(len(annotations))
            else: lens_per_annotator[annotator] = [len(annotations)]
        self._avg_nb_units_per_annotator = {}
        self._std_nb_units_per_annotator = {}
        for annotator in lens_per_annotator.keys():
            nb_units = lens_per_annotator[annotator]
            self._avg_nb_units_per_annotator[annotator] = float(np.mean(nb_units))
            self._std_nb_units_per_annotator[annotator] = float(np.std(nb_units))

    def print_perannot_stats(self):
        for ann in self._annotators():
            print(f'ANNOTATOR: {ann}')
            weights = ','.join([f'{w:.3f}' for w in self._categories_weight[ann]])
            print(f'Categ weights: {weights}')
            print(f'Num. of units: {self._avg_nb_units_per_annotator[ann]:.3f}, {self._std_nb_units_per_annotator[ann]:.3f}')
            print(f'Unit duration: {self._avg_unit_duration[ann]:.3f}, {self._std_unit_duration[ann]:.3f}')
            print(f'Average gap  : {self._avg_gap[ann]:.3f}, {self._std_gap[ann]:.3f}')

    @property
    def sample_from_continuum(self) -> Continuum:
        self._has_been_init()
        new_continnum = self._reference_continuum.copy_flush()
        for annotator in self._annotators():
            new_continnum.add_annotator(annotator)
            last_point = 0
            nb_units = abs(int(np.random.normal(self._avg_nb_units_per_annotator[annotator],
                                                self._std_nb_units_per_annotator[annotator])))
            nb_units = max(1, nb_units) # always generate at least one unit
            for _ in range(nb_units):
                gap = np.random.normal(self._avg_gap[annotator], self._std_gap[annotator])
                start = last_point + gap
                end = start + abs(np.random.normal(self._avg_unit_duration[annotator], self._std_unit_duration[annotator]))
                # Segments shorter than segment precision are illegal for pyannote
                while end - start < pyannote.core.segment.SEGMENT_PRECISION:
                    end = start + abs(np.random.normal(self._avg_unit_duration[annotator], self._std_unit_duration[annotator]))
                category = np.random.choice(self._categories, p=self._categories_weight[annotator])
                new_continnum.add(annotator, Segment(start, end), category)
                last_point = end
        return new_continnum

class StatSamplerPerAnnotatorNumunits(StatisticalContinuumSampler):
    '''
    Calculate span-generating statistic, while calculating 'number of units' stats separately for each annotator.
    Input continuum has annotators names formated like NAME_FILENUM in order to distinguish different files.
    Per-annotator num. units stats are calculated for 'base' annotator names, ie NAME, and
    continuums are generated with these names.
    '''

    def _set_nb_units_information(self):
        lens_per_annotator = {}
        for nannotator, annotations in self._reference_continuum._annotations.items():
            #print(nannotator, len(annotations))
            annotator = nannotator.split('_')[0]
            if annotator in lens_per_annotator: lens_per_annotator[annotator].append(len(annotations))
            else: lens_per_annotator[annotator] = [len(annotations)]
        self._avg_nb_units_per_annotator = {}
        self._std_nb_units_per_annotator = {}
        for annotator in lens_per_annotator.keys():
            nb_units = lens_per_annotator[annotator]
            self._avg_nb_units_per_annotator[annotator] = float(np.mean(nb_units))
            self._std_nb_units_per_annotator[annotator] = float(np.std(nb_units))

    @property
    def sample_from_continuum(self) -> Continuum:
        self._has_been_init()
        new_continnum = self._reference_continuum.copy_flush()
        for annotator in self._ground_truth_annotators:
            new_continnum.add_annotator(annotator)
            last_point = 0
            nb_units = abs(int(np.random.normal(self._avg_nb_units_per_annotator[annotator],
                                                self._std_nb_units_per_annotator[annotator])))
            nb_units = max(1, nb_units) # at least one unit per annotator
            for _ in range(nb_units):
                gap = np.random.normal(self._avg_gap, self._std_gap)
                start = last_point + gap

                end = start + abs(np.random.normal(self._avg_unit_duration, self._std_unit_duration))
                # Segments shorter than segment precision are illegal for pyannote
                while end - start < pyannote.core.segment.SEGMENT_PRECISION:
                    end = start + abs(np.random.normal(self._avg_unit_duration, self._std_unit_duration))

                category = np.random.choice(self._categories, p=self._categories_weight)

                new_continnum.add(annotator, Segment(start, end), category)

                last_point = end
        return new_continnum