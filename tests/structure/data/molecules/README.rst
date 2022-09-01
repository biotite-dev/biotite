Test structures
===============

All test structures which where originally only downloaded in one format
have been translated to the other file formats using openbabel.
In the case of zinc_33 the open babel conformer generation has also been used
to generate a single SD File containing multiple models.
The two multimodel files here are 10000_docked and zinc_33_conformers.
For SD files and MOL files only the single model files that did have meta information 
have been transformed to MOL files as well. Otherwise a single model MOL file
with no meta information is identical to an according SD File.


CYN: Caynide        - Contains negatively charged atom and triple bond
HWB: Cyanidin       - Contains positively charged atom
TYR: Tyrosine       - common amino acid

aspirin_*:          - Aspirin with coordinates either in 2D or 3D.
10000_docked:       - Output from docking the gbt15 molecule with id 10000 to the
                      DNA-PKcs kinase active site.
10000_docked_1:     - First model from the 10000_docked multi model file as a 
                      single model file.
10000_docked_2:     - Second model from the 10000_docked multi model file as a 
                      single model file.                                         
zinc_33:            - A more complex example taken from zinc database:
                        https://zinc20.docking.org/substances/ZINC000000000033/
zinc_33_conformers: - As described above a collection of conformers for the
                      ZINC000000000033 molecule. Has been generated with 
                      obabel using the --conformer flag:
                      https://open-babel.readthedocs.io/en/latest/3DStructureGen/multipleconformers.html    
nu7026:             - A known inhibitor for DNA-PKcs.
nu7026_conformers   - As before openbabel has been used to generated multiple
                      conformers of nu7026 to generate this multi model file.
CO2:                - Carbon Dioxide
BENZ:               - benzene as simple example for xyz, but translated to
                      other file formats
lorazepam:          - A commonly known benzodiazepine medication, used to treat
                      e.g., anxiety or seizures.
HArF:               - Inorganing compound ArHF                
                                
