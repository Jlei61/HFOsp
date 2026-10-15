"""Four-arm live report; preserve the original reducer and historical snapshots."""
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import modeling.scripts.topic4_core_count_optimization.report as report
from modeling.scripts.topic4_core_count_optimization.common import *
if __name__ == '__main__':
    report.COLORS[4] = '#4F8A64'
    report.main()
    review = read(OUT / 'review.json')
    review['core_counts'] = read(OUT / 'plan.json')['core_counts']
    if 'training_endpoints' in review and '4' in review['training_endpoints']:
        endpoints = review['training_endpoints']
        review['four_minus_two_per_restart'] = np.asarray(endpoints['4'], dtype=float) - np.asarray(endpoints['2'], dtype=float)
    write(OUT / 'review.json', review)
    path = OUT / 'figures/README.md'
    if path.exists():
        path.write_text(path.read_text().replace('单、双、三 core', '单、双、三、四 core'))
