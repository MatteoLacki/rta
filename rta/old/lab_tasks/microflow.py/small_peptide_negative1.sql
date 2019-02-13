DROP TABLE IF EXISTS small_peptide;
CREATE TABLE small_peptide
SELECT

p.workflow_index as peptideWFindex,
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