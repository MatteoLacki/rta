DROP TABLE IF EXISTS all_signals4;
CREATE TABLE all_signals4
SELECT
    p.sequence              as sequence,
    p.modifier              as modifier,
    p.type                  as type,
    p.score                 as score,
    q.low_energy_index      as low_energy_index,
    c.cluster_average_index as cluster,
    m.workflow_index        as run,
    m.Mass                  as mass,
    m.Intensity             as intensity,
    m.Z                     as charge,
    m.FWHM                  as FWHM,
    m.RT                    as rt,
    m.RTSD                  as rt_sd,
    m.Mobility              as dt,
    m.LiftOffRT             as LiftOffRT,
    m.InfUpRT               as InfUpRT,
    m.InfDownRT             as InfDownRT,
    m.TouchDownRT           as TouchDownRT
FROM
            peptide         as p
JOIN        query_mass      as q on q.index = p.query_mass_index
RIGHT JOIN  mass_spectrum   as m on q.low_energy_index = m.low_energy_index
LEFT  JOIN  clustered_emrt  as c on c.low_energy_index = m.low_energy_index;

SELECT COUNT(*) FROM all_signals4;