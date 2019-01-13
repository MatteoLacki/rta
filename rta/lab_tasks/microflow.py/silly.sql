SELECT COUNT(DISTINCT `low_energy_index`) FROM `clustered_emrt` as  c WHERE c.mass >= 500 and  c.intensity >= 500;;
SELECT COUNT(DISTINCT `low_energy_index`) FROM `mass_spectrum` as  m WHERE m.mass >= 500 and  m.intensity >= 500;
SELECT COUNT(DISTINCT `index`) FROM `low_energy` as  l WHERE l.mass >= 500 and  l.intensity >= 500;





SELECT COUNT(DISTINCT `low_energy_index`) FROM `clustered_emrt` as  c WHERE c.mass >= 500 and  c.inten >= 500;;
SELECT COUNT(DISTINCT `low_energy_index`) FROM `mass_spectrum` as  m WHERE m.Mass >= 500 and  m.Intensity >= 500;
SELECT COUNT(DISTINCT `index`) FROM `low_energy` as  l WHERE l.mass >= 500 ;

SELECT COUNT(DISTINCT `low_energy_index`) FROM `clustered_emrt` as  c WHERE c.mass >= 500 and  c.inten >= 500;;
SELECT COUNT(DISTINCT `low_energy_index`) FROM `mass_spectrum` as  m WHERE m.Mass >= 500 and  m.Intensity >= 500;
SELECT COUNT(DISTINCT `index`) FROM `low_energy` as  l WHERE l.mass >= 500 ;



DROP TABLE IF EXISTS clustered_mass_spectrum;
CREATE TABLE clustered_mass_spectrum
SELECT
	ms.*,
    ce.workflow_index AS ce_wi,
    ce.cluster_average_index AS ce_cai,
    ce.mass AS ce_mass,
    ce.inten AS ce_intensity
FROM mass_spectrum AS ms
JOIN clustered_emrt AS ce USING(low_energy_index);



SELECT COUNT(DISTINCT `low_energy_index`) FROM `clustered_emrt`;
SELECT COUNT(DISTINCT `low_energy_index`) FROM `mass_spectrum`;
SELECT COUNT(DISTINCT `index`) FROM `low_energy`

CREATE TABLE all_signals3
SELECT A.*,  B.cluster_average_index, B.low_energy_index as B_low_energy_index FROM all_signals2 as A LEFT JOIN clustered_emrt as B USING(low_energy_index);