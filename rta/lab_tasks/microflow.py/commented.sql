# this cuts the mass_spectrum row number

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


# this cuts the mass_spectrum row number

DROP TABLE IF EXISTS clustered_mass_spectrum;
CREATE TABLE clustered_mass_spectrum
SELECT
	ms.*,
    ce.workflow_index AS ce_wi,
    ce.cluster_average_index AS ce_cai,
    ce.mass AS ce_mass,
    ce.inten AS ce_intensity
FROM mass_spectrum AS ms
LEFT JOIN clustered_emrt AS ce USING(low_energy_index);