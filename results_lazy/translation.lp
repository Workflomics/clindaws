taxonomy("ape", "Type", ("Type", "1995--Angelo-J-E--Ni-Al-H--LAMMPS--ipr1", "AluminiumPotential")).
taxonomy("ape", "Type", ("Type", "1996--Farkas-D--Nb-Ti-Al--LAMMPS--ipr1", "AluminiumPotential")).
taxonomy("ape", "Type", ("Type", "1997--Ackland-G-J--Fe--LAMMPS--ipr1", "IronPotential")).
taxonomy("ape", "Type", ("Type", "1998--Meyer-R--Fe--LAMMPS--ipr1", "IronPotential")).
taxonomy("ape", "Type", ("Type", "3dPlot", "Plot")).
taxonomy("ape", "Type", ("Type", "Aluminium", "Material")).
taxonomy("ape", "Type", ("Type", "AluminiumPotential", "LammpsPotential")).
taxonomy("ape", "Type", ("Type", "Bulk", "Structure")).
taxonomy("ape", "Type", ("Type", "BulkModulus", "MaterialProperty")).
taxonomy("ape", "Type", ("Type", "ChemicalPotential", "MaterialProperty")).
taxonomy("ape", "Type", ("Type", "DFT", "Engine")).
taxonomy("ape", "Type", ("Type", "DefectConcentration", "MaterialProperty")).
taxonomy("ape", "Type", ("Type", "DefectFormationEnergy", "MaterialProperty")).
taxonomy("ape", "Type", ("Type", "Energy", "MaterialProperty")).
taxonomy("ape", "Type", ("Type", "Engine", "Type")).
taxonomy("ape", "Type", ("Type", "FeAl", "Material")).
taxonomy("ape", "Type", ("Type", "GPAW", "DFT")).
taxonomy("ape", "Type", ("Type", "Iron", "Material")).
taxonomy("ape", "Type", ("Type", "IronPotential", "LammpsPotential")).
taxonomy("ape", "Type", ("Type", "Lammps", "Engine")).
taxonomy("ape", "Type", ("Type", "LammpsPotential", "Potential")).
taxonomy("ape", "Type", ("Type", "Material", "Type")).
taxonomy("ape", "Type", ("Type", "MaterialProperty", "Type")).
taxonomy("ape", "Type", ("Type", "MolecularDynamicsOutput", "Type")).
taxonomy("ape", "Type", ("Type", "Murnaghan", "Type")).
taxonomy("ape", "Type", ("Type", "Plot", "Type")).
taxonomy("ape", "Type", ("Type", "Position", "Type")).
taxonomy("ape", "Type", ("Type", "Potential", "Type")).
taxonomy("ape", "Type", ("Type", "Project", "Type")).
taxonomy("ape", "Type", ("Type", "Slab", "Structure")).
taxonomy("ape", "Type", ("Type", "Sphinx", "DFT")).
taxonomy("ape", "Type", ("Type", "Structure", "Type")).
taxonomy("ape", "Type", ("Type", "VASP", "DFT")).
taxonomy("ape", "Format", ("Format", "Array", "Format")).
taxonomy("ape", "Format", ("Format", "Coordinates", "Format")).
taxonomy("ape", "Format", ("Format", "EngineFormat", "Format")).
taxonomy("ape", "Format", ("Format", "Float", "Value")).
taxonomy("ape", "Format", ("Format", "MurnaghanFormat", "Format")).
taxonomy("ape", "Format", ("Format", "PlotFormat", "Format")).
taxonomy("ape", "Format", ("Format", "ProjectFormat", "Format")).
taxonomy("ape", "Format", ("Format", "String", "Format")).
taxonomy("ape", "Format", ("Format", "StructureFormat", "Format")).
taxonomy("ape", "Format", ("Format", "Value", "Format")).
taxonomy("ape", "Format", ("Format", "pyiron_atomistics.atomistics.master.murnaghan.Murnaghan", "MurnaghanFormat")).
taxonomy("ape", "Format", ("Format", "pyiron_atomistics.atomistics.structure.atoms.Atoms", "StructureFormat")).
taxonomy("ape", "Format", ("Format", "pyiron_atomistics.gpaw.gpaw.Gpaw", "EngineFormat")).
taxonomy("ape", "Format", ("Format", "pyiron_atomistics.lammps.lammps.Lammps", "EngineFormat")).
taxonomy("ape", "Format", ("Format", "pyiron_atomistics.project.Project", "ProjectFormat")).
taxonomy("ape", "Format", ("Format", "pyiron_atomistics.sphinx.sphinx.Sphinx", "EngineFormat")).
taxonomy("ape", "Format", ("Format", "pyiron_atomistics.vasp.vasp.Vasp", "EngineFormat")).
tool_taxonomy("ape", ("CalcChemicalPotentialA", "GetProperty")).
tool_taxonomy("ape", ("CalcChemicalPotentialB", "GetProperty")).
tool_taxonomy("ape", ("CalcMD", "ConfigureJob")).
tool_taxonomy("ape", ("CalculateDefectConcentration", "GetProperty")).
tool_taxonomy("ape", ("CalculateDefectFormationEnergy", "GetProperty")).
tool_taxonomy("ape", ("ConfigureJob", "Tool")).
tool_taxonomy("ape", ("ConfigureStructure", "Tool")).
tool_taxonomy("ape", ("CreateAntisite", "CreateDefect")).
tool_taxonomy("ape", ("CreateAntisiteAl", "CreateAntisite")).
tool_taxonomy("ape", ("CreateAntisiteFe", "CreateAntisite")).
tool_taxonomy("ape", ("CreateDefect", "ConfigureStructure")).
tool_taxonomy("ape", ("CreateJob", "Tool")).
tool_taxonomy("ape", ("CreateJobGpaw", "CreateJob")).
tool_taxonomy("ape", ("CreateJobLammps", "CreateJob")).
tool_taxonomy("ape", ("CreateJobMurnaghan", "CreateJob")).
tool_taxonomy("ape", ("CreateJobSphinx", "CreateJob")).
tool_taxonomy("ape", ("CreateJobVasp", "CreateJob")).
tool_taxonomy("ape", ("CreateProject", "Tool")).
tool_taxonomy("ape", ("CreateStructure", "Tool")).
tool_taxonomy("ape", ("CreateStructureBulk", "CreateStructure")).
tool_taxonomy("ape", ("CreateStructureSlab", "CreateStructure")).
tool_taxonomy("ape", ("CreateVacancy", "CreateDefect")).
tool_taxonomy("ape", ("CreateVacancyAl", "CreateVacancy")).
tool_taxonomy("ape", ("CreateVacancyFe", "CreateVacancy")).
tool_taxonomy("ape", ("GetBulkModulus", "GetProperty")).
tool_taxonomy("ape", ("GetMDOutput", "GetOutput")).
tool_taxonomy("ape", ("GetOutput", "Tool")).
tool_taxonomy("ape", ("GetProperty", "Tool")).
tool_taxonomy("ape", ("PlotStructure", "Plotting")).
tool_taxonomy("ape", ("Plotting", "Tool")).
tool_taxonomy("ape", ("RelaxStructure", "ConfigureStructure")).
tool_taxonomy("ape", ("RunDFT", "RunJob")).
tool_taxonomy("ape", ("RunGpaw", "RunDFT")).
tool_taxonomy("ape", ("RunJob", "Tool")).
tool_taxonomy("ape", ("RunLammpsJob", "RunJob")).
tool_taxonomy("ape", ("RunMurnaghanJob", "RunJob")).
tool_taxonomy("ape", ("RunSphinx", "RunDFT")).
tool_taxonomy("ape", ("RunVasp", "RunDFT")).
tool_taxonomy("ape", ("SelectPotential", "ConfigureJob")).
tool_taxonomy("ape", ("SetReferenceJob", "ConfigureJob")).
tool("create_project").
tool_taxonomy("ape", ("create_project", "CreateProject")).
tool("create_structure_bulk").
tool_taxonomy("ape", ("create_structure_bulk", "CreateStructureBulk")).
tool("create_structure_slab").
tool_taxonomy("ape", ("create_structure_slab", "CreateStructureSlab")).
tool("plot_structure").
tool_taxonomy("ape", ("plot_structure", "PlotStructure")).
tool("create_vacancy_Al").
tool_taxonomy("ape", ("create_vacancy_Al", "CreateVacancyAl")).
tool("create_vacancy_Fe").
tool_taxonomy("ape", ("create_vacancy_Fe", "CreateVacancyFe")).
tool("create_antisite_Al").
tool_taxonomy("ape", ("create_antisite_Al", "CreateAntisiteAl")).
tool("create_antisite_Fe").
tool_taxonomy("ape", ("create_antisite_Fe", "CreateAntisiteFe")).
tool("relax_structure__ann0").
tool_taxonomy("ape", ("relax_structure__ann0", "RelaxStructure")).
tool("relax_structure__ann1").
tool_taxonomy("ape", ("relax_structure__ann1", "RelaxStructure")).
tool("create_job_lammps").
tool_taxonomy("ape", ("create_job_lammps", "CreateJobLammps")).
tool("create_job_gpaw").
tool_taxonomy("ape", ("create_job_gpaw", "CreateJobGpaw")).
tool("create_job_sphinx").
tool_taxonomy("ape", ("create_job_sphinx", "CreateJobSphinx")).
tool("create_job_vasp").
tool_taxonomy("ape", ("create_job_vasp", "CreateJobVasp")).
tool("create_job_murn").
tool_taxonomy("ape", ("create_job_murn", "CreateJobMurnaghan")).
tool("run_job_lammps").
tool_taxonomy("ape", ("run_job_lammps", "RunLammpsJob")).
tool("run_job_murn").
tool_taxonomy("ape", ("run_job_murn", "RunMurnaghanJob")).
tool("calc_md").
tool_taxonomy("ape", ("calc_md", "CalcMD")).
tool("select_potential").
tool_taxonomy("ape", ("select_potential", "SelectPotential")).
tool("run_vasp").
tool_taxonomy("ape", ("run_vasp", "RunVasp")).
tool("run_sphinx").
tool_taxonomy("ape", ("run_sphinx", "RunSphinx")).
tool("run_gpaw").
tool_taxonomy("ape", ("run_gpaw", "RunGpaw")).
tool("calculate_defect_formation_energy").
tool_taxonomy("ape", ("calculate_defect_formation_energy", "CalculateDefectFormationEnergy")).
tool("calc_defect_concentration").
tool_taxonomy("ape", ("calc_defect_concentration", "CalculateDefectConcentration")).
tool("calc_chemical_potential_A").
tool_taxonomy("ape", ("calc_chemical_potential_A", "CalcChemicalPotentialA")).
tool("calc_chemical_potential_B").
tool_taxonomy("ape", ("calc_chemical_potential_B", "CalcChemicalPotentialB")).
tool("get_property_bulkmodulus").
tool_taxonomy("ape", ("get_property_bulkmodulus", "GetBulkModulus")).
tool("get_output_MD").
tool_taxonomy("ape", ("get_output_MD", "GetMDOutput")).
holds(0, avail("wf_input_0")).
holds(0, dim("wf_input_0", "String", "Format")).
ape_holds_dim("wf_input_0", "String", "Format").
holds(0, dim("wf_input_0", "FeAl", "Type")).
ape_holds_dim("wf_input_0", "FeAl", "Type").
holds(0, avail("wf_input_1")).
holds(0, dim("wf_input_1", "Coordinates", "Format")).
ape_holds_dim("wf_input_1", "Coordinates", "Format").
holds(0, dim("wf_input_1", "Position", "Type")).
ape_holds_dim("wf_input_1", "Position", "Type").
goal_output(0, "Float", "Format").
goal_output(0, "DefectConcentration", "Type").
enable_all_inputs_used.
enable_primary_output_consumed.
exact_horizon_mode.
lazy_tool_candidate("create_project_lc0", "create_project").
lazy_candidate_output_port("create_project_lc0", 0).
lazy_candidate_output_singleton("create_project_lc0", 0, "pyiron_atomistics.project.Project", "Format").
lazy_candidate_output_singleton("create_project_lc0", 0, "Project", "Type").
lazy_tool_candidate("create_structure_bulk_lc0", "create_structure_bulk").
lazy_candidate_input_port("create_structure_bulk_lc0", 0).
lazy_candidate_input_value("create_structure_bulk_lc0", 0, "Project", "Type").
lazy_candidate_input_port("create_structure_bulk_lc0", 1).
lazy_candidate_input_value("create_structure_bulk_lc0", 1, "String", "Format").
lazy_candidate_input_value("create_structure_bulk_lc0", 1, "Aluminium", "Type").
lazy_candidate_input_value("create_structure_bulk_lc0", 1, "FeAl", "Type").
lazy_candidate_input_value("create_structure_bulk_lc0", 1, "Iron", "Type").
lazy_initial_bindable("create_structure_bulk_lc0", 1, "wf_input_0").
lazy_candidate_output_port("create_structure_bulk_lc0", 0).
lazy_candidate_output_singleton("create_structure_bulk_lc0", 0, "pyiron_atomistics.atomistics.structure.atoms.Atoms", "Format").
lazy_candidate_output_singleton("create_structure_bulk_lc0", 0, "Bulk", "Type").
lazy_tool_candidate("create_structure_slab_lc0", "create_structure_slab").
lazy_candidate_input_port("create_structure_slab_lc0", 0).
lazy_candidate_input_value("create_structure_slab_lc0", 0, "Project", "Type").
lazy_candidate_input_port("create_structure_slab_lc0", 1).
lazy_candidate_input_value("create_structure_slab_lc0", 1, "String", "Format").
lazy_candidate_input_value("create_structure_slab_lc0", 1, "Aluminium", "Type").
lazy_candidate_input_value("create_structure_slab_lc0", 1, "FeAl", "Type").
lazy_candidate_input_value("create_structure_slab_lc0", 1, "Iron", "Type").
lazy_initial_bindable("create_structure_slab_lc0", 1, "wf_input_0").
lazy_candidate_output_port("create_structure_slab_lc0", 0).
lazy_candidate_output_singleton("create_structure_slab_lc0", 0, "pyiron_atomistics.atomistics.structure.atoms.Atoms", "Format").
lazy_candidate_output_singleton("create_structure_slab_lc0", 0, "Slab", "Type").
lazy_tool_candidate("plot_structure_lc0", "plot_structure").
lazy_candidate_input_port("plot_structure_lc0", 0).
lazy_candidate_input_value("plot_structure_lc0", 0, "Bulk", "Type").
lazy_candidate_input_value("plot_structure_lc0", 0, "Slab", "Type").
lazy_candidate_output_port("plot_structure_lc0", 0).
lazy_candidate_output_singleton("plot_structure_lc0", 0, "PlotFormat", "Format").
lazy_candidate_output_singleton("plot_structure_lc0", 0, "3dPlot", "Type").
lazy_tool_candidate("create_vacancy_Al_lc0", "create_vacancy_Al").
lazy_candidate_input_port("create_vacancy_Al_lc0", 0).
lazy_candidate_input_value("create_vacancy_Al_lc0", 0, "Bulk", "Type").
lazy_candidate_input_port("create_vacancy_Al_lc0", 1).
lazy_candidate_input_value("create_vacancy_Al_lc0", 1, "Coordinates", "Format").
lazy_candidate_input_value("create_vacancy_Al_lc0", 1, "Position", "Type").
lazy_initial_bindable("create_vacancy_Al_lc0", 1, "wf_input_1").
lazy_candidate_output_port("create_vacancy_Al_lc0", 0).
lazy_candidate_output_singleton("create_vacancy_Al_lc0", 0, "pyiron_atomistics.atomistics.structure.atoms.Atoms", "Format").
lazy_candidate_output_singleton("create_vacancy_Al_lc0", 0, "Bulk", "Type").
lazy_tool_candidate("create_vacancy_Fe_lc0", "create_vacancy_Fe").
lazy_candidate_input_port("create_vacancy_Fe_lc0", 0).
lazy_candidate_input_value("create_vacancy_Fe_lc0", 0, "Bulk", "Type").
lazy_candidate_input_port("create_vacancy_Fe_lc0", 1).
lazy_candidate_input_value("create_vacancy_Fe_lc0", 1, "Coordinates", "Format").
lazy_candidate_input_value("create_vacancy_Fe_lc0", 1, "Position", "Type").
lazy_initial_bindable("create_vacancy_Fe_lc0", 1, "wf_input_1").
lazy_candidate_output_port("create_vacancy_Fe_lc0", 0).
lazy_candidate_output_singleton("create_vacancy_Fe_lc0", 0, "pyiron_atomistics.atomistics.structure.atoms.Atoms", "Format").
lazy_candidate_output_singleton("create_vacancy_Fe_lc0", 0, "Bulk", "Type").
lazy_tool_candidate("create_antisite_Al_lc0", "create_antisite_Al").
lazy_candidate_input_port("create_antisite_Al_lc0", 0).
lazy_candidate_input_value("create_antisite_Al_lc0", 0, "Bulk", "Type").
lazy_candidate_input_port("create_antisite_Al_lc0", 1).
lazy_candidate_input_value("create_antisite_Al_lc0", 1, "Coordinates", "Format").
lazy_candidate_input_value("create_antisite_Al_lc0", 1, "Position", "Type").
lazy_initial_bindable("create_antisite_Al_lc0", 1, "wf_input_1").
lazy_candidate_output_port("create_antisite_Al_lc0", 0).
lazy_candidate_output_singleton("create_antisite_Al_lc0", 0, "pyiron_atomistics.atomistics.structure.atoms.Atoms", "Format").
lazy_candidate_output_singleton("create_antisite_Al_lc0", 0, "Bulk", "Type").
lazy_tool_candidate("create_antisite_Fe_lc0", "create_antisite_Fe").
lazy_candidate_input_port("create_antisite_Fe_lc0", 0).
lazy_candidate_input_value("create_antisite_Fe_lc0", 0, "Bulk", "Type").
lazy_candidate_input_port("create_antisite_Fe_lc0", 1).
lazy_candidate_input_value("create_antisite_Fe_lc0", 1, "Coordinates", "Format").
lazy_candidate_input_value("create_antisite_Fe_lc0", 1, "Position", "Type").
lazy_initial_bindable("create_antisite_Fe_lc0", 1, "wf_input_1").
lazy_candidate_output_port("create_antisite_Fe_lc0", 0).
lazy_candidate_output_singleton("create_antisite_Fe_lc0", 0, "pyiron_atomistics.atomistics.structure.atoms.Atoms", "Format").
lazy_candidate_output_singleton("create_antisite_Fe_lc0", 0, "Bulk", "Type").
lazy_tool_candidate("relax_structure__ann0_lc0", "relax_structure__ann0").
lazy_candidate_input_port("relax_structure__ann0_lc0", 0).
lazy_candidate_input_value("relax_structure__ann0_lc0", 0, "Bulk", "Type").
lazy_candidate_output_port("relax_structure__ann0_lc0", 0).
lazy_candidate_output_singleton("relax_structure__ann0_lc0", 0, "pyiron_atomistics.atomistics.structure.atoms.Atoms", "Format").
lazy_candidate_output_singleton("relax_structure__ann0_lc0", 0, "Bulk", "Type").
lazy_tool_candidate("relax_structure__ann1_lc0", "relax_structure__ann1").
lazy_candidate_input_port("relax_structure__ann1_lc0", 0).
lazy_candidate_input_value("relax_structure__ann1_lc0", 0, "Slab", "Type").
lazy_candidate_output_port("relax_structure__ann1_lc0", 0).
lazy_candidate_output_singleton("relax_structure__ann1_lc0", 0, "pyiron_atomistics.atomistics.structure.atoms.Atoms", "Format").
lazy_candidate_output_singleton("relax_structure__ann1_lc0", 0, "Slab", "Type").
lazy_tool_candidate("create_job_lammps_lc0", "create_job_lammps").
lazy_candidate_input_port("create_job_lammps_lc0", 0).
lazy_candidate_input_value("create_job_lammps_lc0", 0, "Project", "Type").
lazy_candidate_input_port("create_job_lammps_lc0", 1).
lazy_candidate_input_value("create_job_lammps_lc0", 1, "Bulk", "Type").
lazy_candidate_input_value("create_job_lammps_lc0", 1, "Slab", "Type").
lazy_candidate_output_port("create_job_lammps_lc0", 0).
lazy_candidate_output_singleton("create_job_lammps_lc0", 0, "pyiron_atomistics.lammps.lammps.Lammps", "Format").
lazy_candidate_output_singleton("create_job_lammps_lc0", 0, "Lammps", "Type").
lazy_tool_candidate("create_job_gpaw_lc0", "create_job_gpaw").
lazy_candidate_input_port("create_job_gpaw_lc0", 0).
lazy_candidate_input_value("create_job_gpaw_lc0", 0, "Project", "Type").
lazy_candidate_input_port("create_job_gpaw_lc0", 1).
lazy_candidate_input_value("create_job_gpaw_lc0", 1, "Bulk", "Type").
lazy_candidate_input_value("create_job_gpaw_lc0", 1, "Slab", "Type").
lazy_candidate_output_port("create_job_gpaw_lc0", 0).
lazy_candidate_output_singleton("create_job_gpaw_lc0", 0, "pyiron_atomistics.gpaw.gpaw.Gpaw", "Format").
lazy_candidate_output_singleton("create_job_gpaw_lc0", 0, "GPAW", "Type").
lazy_tool_candidate("create_job_sphinx_lc0", "create_job_sphinx").
lazy_candidate_input_port("create_job_sphinx_lc0", 0).
lazy_candidate_input_value("create_job_sphinx_lc0", 0, "Project", "Type").
lazy_candidate_input_port("create_job_sphinx_lc0", 1).
lazy_candidate_input_value("create_job_sphinx_lc0", 1, "Bulk", "Type").
lazy_candidate_input_value("create_job_sphinx_lc0", 1, "Slab", "Type").
lazy_candidate_output_port("create_job_sphinx_lc0", 0).
lazy_candidate_output_singleton("create_job_sphinx_lc0", 0, "pyiron_atomistics.sphinx.sphinx.Sphinx", "Format").
lazy_candidate_output_singleton("create_job_sphinx_lc0", 0, "Sphinx", "Type").
lazy_tool_candidate("create_job_vasp_lc0", "create_job_vasp").
lazy_candidate_input_port("create_job_vasp_lc0", 0).
lazy_candidate_input_value("create_job_vasp_lc0", 0, "Project", "Type").
lazy_candidate_input_port("create_job_vasp_lc0", 1).
lazy_candidate_input_value("create_job_vasp_lc0", 1, "Bulk", "Type").
lazy_candidate_input_value("create_job_vasp_lc0", 1, "Slab", "Type").
lazy_candidate_output_port("create_job_vasp_lc0", 0).
lazy_candidate_output_singleton("create_job_vasp_lc0", 0, "pyiron_atomistics.vasp.vasp.Vasp", "Format").
lazy_candidate_output_singleton("create_job_vasp_lc0", 0, "VASP", "Type").
lazy_tool_candidate("create_job_murn_lc0", "create_job_murn").
lazy_candidate_input_port("create_job_murn_lc0", 0).
lazy_candidate_input_value("create_job_murn_lc0", 0, "Project", "Type").
lazy_candidate_input_port("create_job_murn_lc0", 1).
lazy_candidate_input_value("create_job_murn_lc0", 1, "GPAW", "Type").
lazy_candidate_input_value("create_job_murn_lc0", 1, "Sphinx", "Type").
lazy_candidate_input_value("create_job_murn_lc0", 1, "VASP", "Type").
lazy_candidate_input_value("create_job_murn_lc0", 1, "Lammps", "Type").
lazy_candidate_output_port("create_job_murn_lc0", 0).
lazy_candidate_output_singleton("create_job_murn_lc0", 0, "pyiron_atomistics.atomistics.master.murnaghan.Murnaghan", "Format").
lazy_candidate_output_singleton("create_job_murn_lc0", 0, "Murnaghan", "Type").
lazy_tool_candidate("run_job_lammps_lc0", "run_job_lammps").
lazy_candidate_input_port("run_job_lammps_lc0", 0).
lazy_candidate_input_value("run_job_lammps_lc0", 0, "Lammps", "Type").
lazy_candidate_output_port("run_job_lammps_lc0", 0).
lazy_candidate_output_singleton("run_job_lammps_lc0", 0, "pyiron_atomistics.lammps.lammps.Lammps", "Format").
lazy_candidate_output_singleton("run_job_lammps_lc0", 0, "Lammps", "Type").
lazy_tool_candidate("run_job_murn_lc0", "run_job_murn").
lazy_candidate_input_port("run_job_murn_lc0", 0).
lazy_candidate_input_value("run_job_murn_lc0", 0, "Murnaghan", "Type").
lazy_candidate_output_port("run_job_murn_lc0", 0).
lazy_candidate_output_singleton("run_job_murn_lc0", 0, "pyiron_atomistics.atomistics.master.murnaghan.Murnaghan", "Format").
lazy_candidate_output_singleton("run_job_murn_lc0", 0, "Murnaghan", "Type").
lazy_tool_candidate("calc_md_lc0", "calc_md").
lazy_candidate_input_port("calc_md_lc0", 0).
lazy_candidate_input_value("calc_md_lc0", 0, "Lammps", "Type").
lazy_candidate_output_port("calc_md_lc0", 0).
lazy_candidate_output_singleton("calc_md_lc0", 0, "pyiron_atomistics.lammps.lammps.Lammps", "Format").
lazy_candidate_output_singleton("calc_md_lc0", 0, "Lammps", "Type").
lazy_tool_candidate("select_potential_lc0", "select_potential").
lazy_candidate_input_port("select_potential_lc0", 0).
lazy_candidate_input_value("select_potential_lc0", 0, "Lammps", "Type").
lazy_candidate_output_port("select_potential_lc0", 0).
lazy_candidate_output_singleton("select_potential_lc0", 0, "pyiron_atomistics.lammps.lammps.Lammps", "Format").
lazy_candidate_output_singleton("select_potential_lc0", 0, "Lammps", "Type").
lazy_tool_candidate("run_vasp_lc0", "run_vasp").
lazy_candidate_input_port("run_vasp_lc0", 0).
lazy_candidate_input_value("run_vasp_lc0", 0, "Bulk", "Type").
lazy_candidate_input_value("run_vasp_lc0", 0, "Slab", "Type").
lazy_candidate_output_port("run_vasp_lc0", 0).
lazy_candidate_output_singleton("run_vasp_lc0", 0, "Float", "Format").
lazy_candidate_output_singleton("run_vasp_lc0", 0, "Energy", "Type").
lazy_tool_candidate("run_sphinx_lc0", "run_sphinx").
lazy_candidate_input_port("run_sphinx_lc0", 0).
lazy_candidate_input_value("run_sphinx_lc0", 0, "Bulk", "Type").
lazy_candidate_input_value("run_sphinx_lc0", 0, "Slab", "Type").
lazy_candidate_output_port("run_sphinx_lc0", 0).
lazy_candidate_output_singleton("run_sphinx_lc0", 0, "Float", "Format").
lazy_candidate_output_singleton("run_sphinx_lc0", 0, "Energy", "Type").
lazy_tool_candidate("run_gpaw_lc0", "run_gpaw").
lazy_candidate_input_port("run_gpaw_lc0", 0).
lazy_candidate_input_value("run_gpaw_lc0", 0, "Bulk", "Type").
lazy_candidate_input_value("run_gpaw_lc0", 0, "Slab", "Type").
lazy_candidate_output_port("run_gpaw_lc0", 0).
lazy_candidate_output_singleton("run_gpaw_lc0", 0, "Float", "Format").
lazy_candidate_output_singleton("run_gpaw_lc0", 0, "Energy", "Type").
lazy_tool_candidate("calculate_defect_formation_energy_lc0", "calculate_defect_formation_energy").
lazy_candidate_input_port("calculate_defect_formation_energy_lc0", 0).
lazy_candidate_input_value("calculate_defect_formation_energy_lc0", 0, "Energy", "Type").
lazy_candidate_input_port("calculate_defect_formation_energy_lc0", 1).
lazy_candidate_input_value("calculate_defect_formation_energy_lc0", 1, "Energy", "Type").
lazy_candidate_output_port("calculate_defect_formation_energy_lc0", 0).
lazy_candidate_output_singleton("calculate_defect_formation_energy_lc0", 0, "Float", "Format").
lazy_candidate_output_singleton("calculate_defect_formation_energy_lc0", 0, "DefectFormationEnergy", "Type").
lazy_tool_candidate("calc_defect_concentration_lc0", "calc_defect_concentration").
lazy_candidate_input_port("calc_defect_concentration_lc0", 0).
lazy_candidate_input_value("calc_defect_concentration_lc0", 0, "Float", "Format").
lazy_candidate_input_value("calc_defect_concentration_lc0", 0, "DefectFormationEnergy", "Type").
lazy_candidate_input_port("calc_defect_concentration_lc0", 1).
lazy_candidate_input_value("calc_defect_concentration_lc0", 1, "Float", "Format").
lazy_candidate_input_value("calc_defect_concentration_lc0", 1, "ChemicalPotential", "Type").
lazy_candidate_output_port("calc_defect_concentration_lc0", 0).
lazy_candidate_output_singleton("calc_defect_concentration_lc0", 0, "Float", "Format").
lazy_candidate_output_singleton("calc_defect_concentration_lc0", 0, "DefectConcentration", "Type").
lazy_tool_candidate("calc_chemical_potential_A_lc0", "calc_chemical_potential_A").
lazy_candidate_input_port("calc_chemical_potential_A_lc0", 0).
lazy_candidate_input_value("calc_chemical_potential_A_lc0", 0, "Aluminium", "Type").
lazy_candidate_input_value("calc_chemical_potential_A_lc0", 0, "FeAl", "Type").
lazy_candidate_input_value("calc_chemical_potential_A_lc0", 0, "Iron", "Type").
lazy_initial_bindable("calc_chemical_potential_A_lc0", 0, "wf_input_0").
lazy_candidate_output_port("calc_chemical_potential_A_lc0", 0).
lazy_candidate_output_singleton("calc_chemical_potential_A_lc0", 0, "Float", "Format").
lazy_candidate_output_singleton("calc_chemical_potential_A_lc0", 0, "ChemicalPotential", "Type").
lazy_tool_candidate("calc_chemical_potential_B_lc0", "calc_chemical_potential_B").
lazy_candidate_input_port("calc_chemical_potential_B_lc0", 0).
lazy_candidate_input_value("calc_chemical_potential_B_lc0", 0, "ChemicalPotential", "Type").
lazy_candidate_input_port("calc_chemical_potential_B_lc0", 1).
lazy_candidate_input_value("calc_chemical_potential_B_lc0", 1, "Aluminium", "Type").
lazy_candidate_input_value("calc_chemical_potential_B_lc0", 1, "FeAl", "Type").
lazy_candidate_input_value("calc_chemical_potential_B_lc0", 1, "Iron", "Type").
lazy_initial_bindable("calc_chemical_potential_B_lc0", 1, "wf_input_0").
lazy_candidate_output_port("calc_chemical_potential_B_lc0", 0).
lazy_candidate_output_singleton("calc_chemical_potential_B_lc0", 0, "Float", "Format").
lazy_candidate_output_singleton("calc_chemical_potential_B_lc0", 0, "ChemicalPotential", "Type").
lazy_tool_candidate("get_property_bulkmodulus_lc0", "get_property_bulkmodulus").
lazy_candidate_input_port("get_property_bulkmodulus_lc0", 0).
lazy_candidate_input_value("get_property_bulkmodulus_lc0", 0, "Murnaghan", "Type").
lazy_candidate_output_port("get_property_bulkmodulus_lc0", 0).
lazy_candidate_output_singleton("get_property_bulkmodulus_lc0", 0, "Float", "Format").
lazy_candidate_output_singleton("get_property_bulkmodulus_lc0", 0, "BulkModulus", "Type").
lazy_tool_candidate("get_output_MD_lc0", "get_output_MD").
lazy_candidate_input_port("get_output_MD_lc0", 0).
lazy_candidate_input_value("get_output_MD_lc0", 0, "Lammps", "Type").
lazy_candidate_output_port("get_output_MD_lc0", 0).
lazy_candidate_output_singleton("get_output_MD_lc0", 0, "Array", "Format").
lazy_candidate_output_singleton("get_output_MD_lc0", 0, "MolecularDynamicsOutput", "Type").
lazy_goal_bindable_output("calculate_defect_formation_energy_lc0", 0, "calc_defect_concentration_lc0", 0).
lazy_goal_bindable_output("calc_chemical_potential_A_lc0", 0, "calc_defect_concentration_lc0", 1).
lazy_goal_bindable_output("calc_chemical_potential_B_lc0", 0, "calc_defect_concentration_lc0", 1).
