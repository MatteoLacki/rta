import pandas as pandas
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 100)
from pathlib import Path

data = Path("~/Projects/rta/rta/data/").expanduser()
A = pd.read_csv(data/'annotated_data.csv')
A.to_msgpack(data/"A_zlib.msg", compress='zlib')
A.to_msgpack(data/"A.msg")

U = pd.read_csv(data/'unannotated_data.csv')
U = U.drop(['sequence', 'modification', 'type', 'score'], 1)
U.to_csv(data/'unannotated_data.csv', index=False)
U.to_msgpack(data/"U_zlib.msg", compress='zlib')
U.to_msgpack(data/"U.msg")
