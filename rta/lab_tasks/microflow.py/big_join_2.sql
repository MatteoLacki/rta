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

DROP TABLE IF EXISTS signals_and_clusters;
CREATE TABLE signals_and_clusters
SELECT
	ce.cluster_average_index AS cluster,
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
    m.TouchDownRT   as TouchDownRT
FROM mass_spectrum AS m
LEFT JOIN clustered_emrt AS ce
ON m.index = ce.spec_index;
ALTER TABLE signals_and_clusters ADD INDEX(low_energy_index);

DROP TABLE IF EXISTS all_signals5;
CREATE TABLE all_signals5
SELECT * 
FROM signals_and_clusters
LEFT JOIN small_peptide USING(low_energy_index)


