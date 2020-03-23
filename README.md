# tauTCC
Tensor co-clustering algorithm, based on the optimization of Goodman and Kruskal's tau function.

The main file, containing the code of the algorithm, is 'coclust_3D_tau.py', in folder 'algorithms'. 

The following files apply tauTCC co-clustering algorithm to synthetic and real-word datasets:

- CoClust_3D_DBLP: run tauTCC on the DBLP Four-Area dataset, saved in 'resources/four_area'
	
	Use the following code:
		python CoClust_3D_DBLP.py algorithm level
	where:
		algorithm: string. Optional.
            Accepted values: ALT, AVG, AGG, ALT2, AGG2.
            Default value: ALT2
        level: string. Optional.
            Accepted values: DEBUG, INFO, WARNING, ERROR, CRITICAL
            Default value: WARNING
- CoClust_3D_MovieLens: run tauTCC on one of the two subsets of MovieLens saved in 'resources/MovieLens'.
	Use the following code:
		python CoClust_3D_MovieLens.py tensor algorithm level
	where:
		tensor: integer.
            Accepted values: 1 or 2. 
			1 is a tensor containing movies with genres 'Animation', 'Horror' or 'Documentary';
			2 is a tensor containing movies with genres 'Adventure', 'Comedy' or 'Drama';
        algorithm: string. Optional.
            Accepted values: ALT, AVG, AGG, ALT2, AGG2. Default: ALT2
        level: string. Optional.
            Accepted values: DEBUG, INFO, WARNING, ERROR, CRITICAL
            Default value: WARNING
- CoClust_3D_yelp: run tauTCC on one of the two subsets of yelp datasets, saved in 'resources/yelp'.
	Use the following code:
		python CoClust_3D_yelp.py tensor algorithm level
	where:
		tensor: string.
            Accepted values: TOR or PGH.
			TOR is a tensor containing reviews about restaurants of the city of Toronto
			PGH is a tensor containing reviews about restaurants of the city of Pittsburgh
        algorithm: string. Optional.
            Accepted values: ALT, AVG, AGG, ALT2, AGG2. Default: ALT2
        level: string. Optional.
            Accepted values: DEBUG, INFO, WARNING, ERROR, CRITICAL
            Default value: WARNING
- CoClust_3D_Synth: run tauTCC on synthetic boolean tensors. 
	Use the following code:
		python CoClust_3D_Synth.py n_test dimensions clusters noise algorithm
	where:
		n_test: integer. how many times the test has to be repeted
        dimensions: list of tuples, separated by '-'. Ex: [(100,100,30)-(50,50,50)].
			Each tuple is the shape of a tensor.
        clusters: list of tuples, separated by '-'.
			Each tuple represents the number of embedded clusters on each mode of the tensor
        noise: float between 0 and 1. 
			The level of noise to be added to the perfect block tensors.
        algorithm: string. One of {'ALT', 'AVG', 'AGG', 'ALT2', 'AGG2'}
