from rta.misc import csv2msg
from rta.read.csvs import vars_unlabelled, vars_annotated

csv2msg('/Users/matteo/Projects/rta/data/unlabelled_all.csv', vars_unlabelled)
csv2msg('/Users/matteo/Projects/rta/data/annotated_all.csv', vars_annotated)
