# evacsim
Scenario generator for evacuation planning in the event of wildfire


Requires numpy and matplotlib

INSTALL:

1/ install networkx
pip install networkx
	
2/ install quadtree
python setup.py install [--prefix "some place where PYTHONPATH points to and where you have write permissions"] 
	
	
Example:

python generator.py --road test --printroad
python generator.py test --evacuation --printfire


FORMAT:

n m
population_1 maximum_rate_1 duedate_1
	...
population_n maximum_rate_n duedate_n
capacity_1 k_1 i_1_1 offset_i_1_1 ... i_k_1 offset_i_k_1
	...
capacity_m k_m i_1_m offset_i_1_m ... i_k_m offset_i_k_m