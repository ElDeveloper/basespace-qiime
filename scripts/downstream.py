#!/usr/bin/env python

import sys
import json
import os

import qiime2 as q2
import numpy as np
import qiime2.plugins.diversity.actions as q2_diversity
import qiime2.plugins.taxa.actions as q2_taxa

from biom import Table


def main():

    # TODO: confirm the size of the AMI is fixed to 12 cores, if this
    # is the case then lets always split over 11 jobs
    n_jobs = 11

    with open('/data/input/AppSession.json', 'U') as fd_json:
        app = json.load(fd_json)

    # get command attributes, etc
    for item in app['Properties']['Items']:
        if item['Name'] == 'Input.Projects':
            project_id = item['Items'][0]['Id']
        if item['Name'] == 'Input.rarefaction-depth':
            sampling_depth = int(item['Content'])
        if item['Name'] == 'Input.metadata-name':
            metadata_name = item['Content']

    # from BaseSpace's documentation
    # TODO: is this the path where the data is going to be found
    input_dir = '/data/input/appresults/'

    # TODO: is this the path to save the data to?
    base = os.path.join('/data/output/appresults/', project_id)
    output_dir = os.path.join(base, 'upstream-results')

    os.makedirs(output_dir, exist_ok=True)

    # TODO: include the path to metadata.tsv
    metadata = q2.Metadata.load(os.path.join(base, metadata_name))
    table = q2.Artifact.load(os.path.join(input_dir, 'feature-table.qza'))
    tree = q2.Artifact.load(os.path.join(input_dir, 'rooted-tree.qza'))
    taxonomy = q2.Artifact.load(os.path.join(input_dir, 'taxonomy.qza'))

    # TODO: Add a validation step in the app's form to require at least 3
    # samples
    bt = table.view(Table)
    _, samples = bt.shape

    # if we can't split the UniFrac compute in batches of at least 4 samples
    # then there's no point in parallelizing the analysis
    if (samples // n_jobs) < 4:
        n_jobs = 1

    counts = bt.sum(axis='sample')
    if np.all(sampling_depth > counts):
        raise ValueError("The selected rarefaction depth removes all "
                         "samples from the analysis. Please use a "
                         "different value.")


    # TODO: Figure out the number of jobs we need. If this is running in
    # a table with few samples don't even parallelize and maybe warn about the
    # one sample case from the form.
    diversity_res = q2_diversity.core_metrics_phylogenetic(table=table,
            metadata=metadata, sampling_depth=sampling_depth, phylogeny=tree)

    # save all the results in a new directory
    output = os.path.join(output_dir, 'core-diversity-analyses')
    os.makedirs(output, exist_ok=True)
    for artifact, name in zip(diversity_res, diversity_res._fields):
        artifact.save(os.path.join(output, name))

    # TODO: Some of these parameters should probably be listed in the
    # user interface
    res = q2_diversity.alpha_rarefaction(table,
                                   min_depth=int(0.10 * sampling_depth),
                                   max_depth=sampling_depth,
                                   phylogeny=tree,
                                   metadata=metadata)
    res.save('alpha-rarefaction')

    res = q2_taxa.barplot(table=table, taxonomy=taxonomy, metadata=metadata)
    res.save('taxonomic-barplot')

    return 0


if __name__ == '__main__':
    sys.exit(main())
