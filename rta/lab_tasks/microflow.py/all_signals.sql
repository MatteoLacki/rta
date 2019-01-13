DROP TABLE IF EXISTS small_peptide;
CREATE TABLE small_peptide
SELECT
    p.sequence      as sequence,
    p.modifier      as modifier,
    p.type          as type,
    p.score         as score,
    q.low_energy_index as low_energy_index
FROM
    peptide as p
    JOIN query_mass as q on q.index = p.query_mass_index
GROUP BY q.low_energy_index;

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
    left join small_peptide as p USING(low_energy_index);

# DROP TABLE IF EXISTS small_peptide;