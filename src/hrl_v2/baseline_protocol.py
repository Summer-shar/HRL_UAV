"""Versioned baseline membership and paper labels; no identity exceptions."""


def is_paper_caliber(objective):
    """The physical objective, not the unchanged Lite observation, selects v2.3."""
    return (objective.get('energy_mode') == 'deployment_segment_propulsion'
            and objective.get('propulsion_weight') == .005)


def reference_combinations(config, *, include_greedy=False):
    combinations = {'hover_local': ('hover', 'local'),
                    'hover_equal': ('hover', 'equal'), 'hover_cvx': ('hover', 'cvx')}
    if not is_paper_caliber(config['objective']):
        combinations['centroid_equal'] = ('centroid', 'equal')
    if include_greedy:
        combinations['hover_greedy'] = ('hover', 'greedy')
    return combinations


def reference_modes_complete(config, modes):
    required = set(reference_combinations(config))
    # Legacy training required the exact four-element set. Paper references
    # may retain diagnostic extras, but those extras do not enter G1.
    return required.issubset(modes) if is_paper_caliber(config['objective']) else required == set(modes)


# Independently trained upper policies for the three fixed-routing baselines.
UPPER_BASELINES = {'NC': ('moving_local', 'local'),
                   'TOO': ('moving_equal', 'equal'), 'TFOO': ('moving_cvx', 'cvx')}
PAPER_COMPARISONS = (
    ('NC', 'moving_local', 'moving', 'local'),
    ('CA', 'hover_equal', 'hover', 'equal'),
    ('TOO', 'moving_equal', 'moving', 'equal'),
    ('TFOO', 'moving_cvx', 'moving', 'cvx'),
    ('Proposed', 'moving', 'moving', None),
)
PAPER_SUPPLEMENTS = (
    (None, 'hover_local', 'hover', 'local'),
    (None, 'shared_upper_equal', 'moving', 'equal'),
    (None, 'shared_upper_cvx', 'moving', 'cvx'),
)


def labeled_results(by_mode):
    """Keep physical metrics intact and label only the surrounding report."""
    main = [dict(paper_name=paper, internal_name=name, metrics=by_mode[name])
            for paper, name, _, _ in PAPER_COMPARISONS if name in by_mode]
    primary = {row['internal_name'] for row in main}
    supplement = [dict(internal_name=name, metrics=metrics)
                  for name, metrics in by_mode.items() if name not in primary]
    return dict(main_comparison=main, supplementary=supplement)
