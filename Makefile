gaussian_mixture_spline_plot: ## Plot of a gaussian mixture model fit.
	../py3_6/bin/python3 ./rta/models/splines/run/gaussian_mixture_model.py

robust_spline_plot: ## Plot of a robust spline fit.
	../py3_6/bin/python3 ./rta/models/splines/run/robust_model.py

pypi: ## deploy (haha, no it won't... need to write it first ;)
	git tag -d GutenTag
	git push gh :refs/tags/GutenTag

py3: ## Activate python3 virtualevn
	source ../py3_6/bin/activate

projectName = rta


# -----------------------------------------------------------
# -----  EVERYTHING BELOW THIS LINE IS NOT IMPORTANT --------
# -----       (Makefile helpers and decoration)      --------
# -----------------------------------------------------------
#
# Decorative Scripts - Do not edit below this line unless you know what it is

.PHONY: help
.DEFAULT_GOAL := help

NO_COLOR    = \033[0m
INDENT      = -30s
BIGINDENT   = -50s
GREEN       = \033[36m
BLUE        = \033[34m
DIM         = \033[2m
help:
	@printf '\n\n$(DIM)Commands:$(NO_COLOR)\n'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN) % $(BIGINDENT)$(NO_COLOR) - %s\n", $$1, $$2}'
