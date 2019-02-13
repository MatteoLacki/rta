DROP TABLE IF EXISTS small_peptide;
CREATE TABLE small_peptide
SELECT

p.workflow_index,
p.query_mass_index as peptideQMindex,
p.sequence      as sequence,
p.modifier      as modifier,
p.type             as type,
p.score            as score,
q.low_energy_index as low_energy_index,
q.index as qmi,
q.workflow_index as qwi

FROM
peptide as p
join query_mass as q on q.index = p.query_mass_index and p.workflow_index=q.workflow_index
GROUP BY q.low_energy_index, qwi;

ALTER TABLE small_peptide ADD INDEX(low_energy_index);

DROP TABLE IF EXISTS all_signals;
CREATE TABLE all_signals
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
    m.InfDownRT     as InfDownRT,
    m.TouchDownRT   as TouchDownRT,
    p.sequence      as sequence,
    p.modifier      as modification,
    p.type          as type,
    p.score         as score
FROM 
    mass_spectrum as m
    left join small_peptide as p USING(low_energy_index, workflow_index);

DROP TABLE IF EXISTS small_peptide;

D%3A%5Cdata%5Cplgs302b5%5Cprojects%5C2018-04_Microflow_final_runs_reprocessed%5Croot
D%3A%5Cdata%5Cplgs302b5%5Cprojects%5C2018-04_Microflow_60min%5Croot
