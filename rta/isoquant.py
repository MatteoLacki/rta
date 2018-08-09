import pandas as pd
import pymysql
from sqlalchemy import create_engine


def drop_table_sql(name = "small_peptide"):
    """Generate an SQL statement for droping a table."""
    return f"DROP TABLE IF EXISTS {name};"

def retrieve_data(password, 
                  project,
                  user,
                  ip,
                  help_tbl = "small_peptide",
                  sign_tbl = "all_signals",
                  verbose  = False,
                  metadata = True):
    """Retrieve data for retention time alignment.

    All created tables are temporary and destroyed at the end of the session.

    Args:
        password (str):     Password to the database.
        user     (str):     Name of the user.
        ip       (str):     IP of the host.
        project  (str):     Name of the project to retrieve data from.
        help_tbl (str):     Name of the help table with info on peptides.
        sign_tbl (str):     Name of the table constructed in DB with final outcomes.
    """
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{ip}/{project}",
                           echo=verbose)
    existing_table_names = engine.table_names()
    assert help_tbl not in existing_table_names, "Choose a different name for 'help_tbl'. This software does not allow to drop existing tables."
    assert sign_tbl not in existing_table_names, "Choose a different name for 'sign_tbl'. This software does not allow to drop existing tables."
    with engine.connect() as c:
        c.execute(drop_table_sql(help_tbl))
        c.execute(f"""CREATE TABLE {help_tbl}
        SELECT
            p.sequence      as sequence,
            p.modifier      as modifier,
            p.type          as type,
            p.score         as score,
            q.low_energy_index as low_energy_index
        FROM
            peptide as p
            join query_mass as q on q.index = p.query_mass_index
        GROUP BY q.low_energy_index;""")
        c.execute(f"ALTER TABLE {help_tbl} ADD INDEX(low_energy_index);")
        c.execute(drop_table_sql(sign_tbl))
        c.execute(f"""CREATE TABLE {sign_tbl}
        SELECT
            m.workflow_index as run,
            m.Mass          as mass,
            m.Intensity     as intensity,
            m.Z             as charge,
            m.FWHM          as FWHM,
            m.RT            as rt,
            m.RTSD          as rt_sd,
            m.Mobility      as dt,
            m.LiftOffRT     as LiftOffRT,
            m.InfUpRT       as InfUpRT,
            m.TouchDownRT   as TouchDownRT,
            p.sequence      as sequence,
            p.modifier      as modification,
            p.type          as type,
            p.score         as score
        FROM 
            mass_spectrum as m
            left join {help_tbl} as p USING(low_energy_index);""")
        # retrieving data.
        df = pd.read_sql_query("SELECT * FROM all_signals", engine)
        # cleaning
        c.execute(drop_table_sql(help_tbl))
        c.execute(drop_table_sql(sign_tbl))
        c.close()
    if metadata:
        project_report  = pd.read_sql_query("SELECT * FROM project", engine)
        workflow_report = pd.read_sql_query("SELECT * FROM workflow_report", engine)
        return df, project_report, workflow_report
    else:
        return df