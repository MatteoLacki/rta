-- DROP TABLE IF EXISTS ccc;
-- CREATE TABLE ccc
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
FROM
    mass_spectrum AS m
    JOIN low_energy AS le ON ms.le_id = le.id AND ms.workflow_index = le.workflow_index
   	
GROUP BY m.le_id;

