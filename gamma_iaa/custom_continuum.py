import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Union, TYPE_CHECKING

import numpy as np
from pygamma_agreement.continuum import PrecisionLevel, _compute_best_alignment_job, \
    _compute_soft_alignment_job, _compute_fast_alignment_job, PRECISION_LEVEL, GammaResults, Continuum
from pygamma_agreement.dissimilarity import AbstractDissimilarity
from sortedcontainers import SortedSet

if TYPE_CHECKING:
    from pygamma_agreement.alignment import Alignment
    from pygamma_agreement.sampler import AbstractContinuumSampler, StatisticalContinuumSampler

class ContinuumFlexSample(Continuum):
    """
    Modification of the class pygamma_agreement.continuum.Continuum to enable calculating gamma
        with a custom compute_gamma() method that enables the use of external continuum (different form self)
        for initializing the sampler of random alignments.
        Othwervise the two classes are identical.
    """

    def compute_gamma(self,
                      dissimilarity: Optional['AbstractDissimilarity'] = None,
                      n_samples: int = 30,
                      precision_level: Optional[Union[float, PrecisionLevel]] = None,
                      ground_truth_annotators: Optional[SortedSet] = None,
                      sampler: 'AbstractContinuumSampler' = None,
                      init_sampler = True,
                      fast: bool = False,
                      soft: bool = False) -> 'GammaResults':
        """

        Parameters
        ----------
        init_sampler: if True (default), the class operates as the orig. pygamma_agreement class.
            Else, if it is False - do not init the sampler - it should be pre-initialized with a custom continuum.
            If sampler is None, it will be True regardless of argument value.

        dissimilarity: AbstractDissimilarity, optional
            dissimilarity instance. Used to compute the disorder between units. If not set, it defaults
            to the combined categorical dissimilarity with parameters taken from the java implementation.
        n_samples: optional int
            number of random continuum sampled from this continuum  used to
            estimate the gamma measure
        precision_level: optional float or "high", "medium", "low"
            error percentage of the gamma estimation. If a literal
            precision level is passed (e.g. "medium"), the corresponding numerical
            value will be used (high: 1%, medium: 2%, low : 5%)
        ground_truth_annotators: SortedSet of str
            if set, the random continuua will only be sampled from these
            annotators. This should be used when you want to compare a prediction
            against some ground truth annotation.
        sampler: AbstractContinuumSampler
            Sampler object, which implements a sampling strategy for creating random continuua used
            to calculate the expected disorder. If not set, defaults to the Statistical continuum sampler
        fast:
            Sets the algorithm to the much faster fast-gamma. It's supposed to be less precise than the "canonical"
            algorithm from Mathet 2015, but usually isn't.
            Performance gains and precision are explained in the Performance section of the documentation.
        soft:
            Activate soft-gamma, an alternative measure that uses a slighlty different definition of an
            alignment. For further information, please consult the 'Soft-Gamma' section of the documentation.
            Incompatible with fast-gamma : raises an error if both 'fast' and 'soft' are set to True.
        """
        from pygamma_agreement.dissimilarity import CombinedCategoricalDissimilarity
        if dissimilarity is None:
            dissimilarity = CombinedCategoricalDissimilarity()

        if sampler is None:
            from pygamma_agreement.sampler import StatisticalContinuumSampler
            sampler = StatisticalContinuumSampler()
            init_sampler = True

        if init_sampler: sampler.init_sampling(self, ground_truth_annotators)

        job = _compute_best_alignment_job
        if soft and fast:
            raise NotImplementedError("Fast-gamma and Soft-gamma are not compatible with each other.")
        if soft:
            job = _compute_soft_alignment_job
        # Multiprocessed computation of sample disorder
        if fast:
            job = _compute_fast_alignment_job
            self.measure_best_window_size(dissimilarity)

        # Multithreaded computation of sample disorder
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as p:
            # Launching jobs
            logging.info(f"Starting computation for the best alignment and a batch of {n_samples} random samples...")
            best_alignment_task = p.submit(job,
                                           *(dissimilarity, self))

            result_pool = [
                # Step one : computing the disorders of a batch of random samples from the continuum (done in parallel)
                p.submit(job,
                         *(dissimilarity, sampler.sample_from_continuum))
                for _ in range(n_samples)
            ]
            chance_best_alignments: List[Alignment] = []
            chance_disorders: List[float] = []

            # Obtaining results
            best_alignment = best_alignment_task.result()
            logging.info("Best alignment obtained")
            for i, result in enumerate(result_pool):
                chance_best_alignments.append(result.result())
                logging.info(f"finished computation of random sample dissimilarity {i + 1}/{n_samples}")
                chance_disorders.append(chance_best_alignments[-1].disorder)
            logging.info("done.")

            if precision_level is not None:
                if isinstance(precision_level, str):
                    precision_level = PRECISION_LEVEL[precision_level]
                assert 0 < precision_level < 1.0
                # If the variation of the disorders of the samples si too high, others are generated.
                # taken from subsection 5.3 of the original paper
                # confidence at 95%, i.e., 1.96
                variation_coeff = np.std(chance_disorders) / np.mean(chance_disorders)
                confidence = 1.96
                required_samples = np.ceil((variation_coeff * confidence / precision_level) ** 2).astype(np.int32)
                if required_samples > n_samples:
                    logging.info(f"Computing second batch of {required_samples - n_samples} "
                                 f"because variation was too high.")
                    result_pool = [
                        p.submit(job,
                                 *(dissimilarity, sampler.sample_from_continuum))
                        for _ in range(required_samples - n_samples)
                    ]
                    for i, result in enumerate(result_pool):
                        chance_best_alignments.append(result.result())
                        logging.info(f"finished computation of additionnal random sample dissimilarity "
                                     f"{i + 1}/{required_samples - n_samples}")
                    logging.info("done.")

        return GammaResults(
            best_alignment=best_alignment,
            chance_alignments=chance_best_alignments,
            precision_level=precision_level,
            dissimilarity=dissimilarity
        )