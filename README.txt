
##########################################################
####### This source code is structured as follows: ####### 

redstarts_birds_raw_data.zip:
	contains raw data

redstarts_birds.xlsx: 
	Birds datasets after beeing processed by removing unwanted variables and performing some aggregation

MAR_shape:
	contain the shape file of our study area (Morocco)

wc2.1_2.5m_bio:
	contains bioclimatic data (raster data)

wc2.1_2.5m_elev:
	contains elevation data (raster data)

MCD12Q1:
	contains land cover data (raster data)

Preprocessing_phase1.R: 
	Performing spatial filtering (thinning) 
	Generate pseudo absence data
	Mapping environmental data with species locations

Preprocessing_phase2.py
	some cleaning (e.g. remove missing values)
	Handling outliers
	Dimensionality reduction (feature selection)
	Variables normalization

Training folder (contains 2 folders and 1 python file):
	Single_models.py:
		training the eight single models on 10 folds CV
	Selecting base-learners:
		contains 2 py files (in 'selecting based on diversity.py' file we calculate the diversity and select the ensembles, 
		and in 'selecting based on performance.py' file we rank the single models using BC and select the base-learners)
	Ensembles trainig
		contains 2 files (in 'diversity-based ensembles.py' file we train the diversity-based ensembles on 10-folds CV, 
		and in 'performance-based ensembles.py' we train the performance-based ensembles)
	
Evaluation:
	contains the scripts for SK test and Borda count
	

	
