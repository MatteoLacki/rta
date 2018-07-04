def parse_formula(formula):
	"""Parse the formula.

	Retrieve names of response and control variables.
	"""
	response = formula.split('~')[0].replace(' ','')
	args = formula[formula.find('(')+1:formula.find(')')]
	args = args.replace(" ","").split(",")
	# check for main argument ("x=")
	control = ""
	for arg in filter(lambda x: "x=" in x, args):
		control = arg[arg.find("=")+1:]
	if not control:
		control = args[0]
	return control, response
