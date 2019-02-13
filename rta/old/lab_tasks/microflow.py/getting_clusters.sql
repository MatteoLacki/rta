-- DROP TABLE IF EXISTS aaa;
-- CREATE TABLE aaa
-- SELECT
--     sp.sequence      as sequence,
--     sp.modifier      as modifier,
--     sp.type          as type,
--     sp.score         as score,
--     sp.low_energy_index as low_energy_index,
--     cemrt.cluster_average_index as cai
-- FROM
--     small_peptide as sp
--     LEFT JOIN clustered_emrt as cemrt USING(low_energy_index)
-- GROUP BY q.low_energy_index;

DROP TABLE IF EXISTS bbb;
CREATE TABLE bbb
SELECT
    p.sequence      as sequence,
    p.modifier      as modifier,
    p.type          as type,
    p.score         as score,
    q.low_energy_index as low_energy_index,
    cemrt.cluster_average_index as cai
FROM
    peptide as p
    JOIN query_mass as q on q.index = p.query_mass_index
    LEFT JOIN clustered_emrt as cemrt on q.low_energy_index = cemrt.low_energy_index
GROUP BY q.low_energy_index;

DROP TABLE IF EXISTS all_signals2;
CREATE TABLE all_signals2
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
    p.score         as score,
    p.cai           as cai
FROM 
    mass_spectrum as m
    left join bbb as p USING(low_energy_index);