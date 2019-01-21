# To Do
* consider storing models instead of data.
	* write a method for the calibrator that will simply call the model on the data.
	* store only the models, without the data.
	* this might be problematic, as the plotting of the model depends on these x and y.
* add some easy way to choose different strategies for x-validation strata
	* add strata based only on the counts of runs a peptide is present at
* add a moving-median-denoising-fitting-of-zero-padded-splines


# Done
* is there any good of doing measure concentration on peptide-presence across runs?
    * not so much: relatively poor concentration properties
    * but did additional Stratified CV here


