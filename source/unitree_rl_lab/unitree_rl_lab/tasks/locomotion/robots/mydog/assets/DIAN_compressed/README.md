# DIAN compressed training asset

`DIAN.usd` and the files in `configuration/` were converted with Isaac Lab's
URDF importer from `DIAN/urdf/DIAN_rl_compressed.urdf`. The URDF's low-poly
visual meshes and high-resolution collision meshes were imported separately.
The source meshes and their triangles are retained in the USD. PhysX uses a
convex hull for each collision mesh, so collision calculations do not preserve
concavities in the original mesh.

The conversion leaves fixed joints unmerged so that the four `*_FOOT_Link`
rigid bodies remain available to the training contact sensors. It uses a free
base and zero imported joint-drive stiffness and damping; the training task's
actuator configuration supplies its own gains. The asset is self-contained
through its relative `configuration/` references.

The `mydog` task selects this asset by default in `robot_cfg.py`. An existing
`DIAN_USD_PATH` environment variable still overrides that default.
