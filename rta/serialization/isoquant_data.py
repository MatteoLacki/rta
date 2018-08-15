import pyarrow          as pa
import pyarrow.parquet  as pq
import os


def dump_retieved(DATA, folder, compression='brotli'):
    """Dump reterieved data."""
    for k, v in zip(('data', 'proj_rep', 'workflow_rep'), DATA):
        os.makedirs(folder, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(v),
                       os.path.join(folder, k + f'.{compression}'),
                       compression = compression)


def load_retrieved(folder):
    """Load data, project report, and workflow report."""
    names_to_ext = dict(x.split('.') for x in os.listdir(folder))
    names = ('data', 'proj_rep', 'workflow_rep')
    return tuple([pq.read_pandas(os.path.join(folder, f"{n}.{names_to_ext[n]}")).to_pandas()
                  for n in names])
