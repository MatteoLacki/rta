# the x-validation has to provide a flow of problems
# but so does the problem itself.

# problem
# --> subproblem -> X-validate
# --> subproblem -> X-validate
# --> subproblem -> X-validate

# all done with one pool of workers
# the x-validation should be a function accepting a model.

def cv_single_process(model, data, params, folds_params):
    errors = {}
    for param in params:
        error = 0.0
        for fold in folds(data, folds_params):
            m = model.fit(fold)
            error += sum(abs(residuals(m)))
        errors[param] = error

    param_opt, error_min = min(errors)
    return param_opt, error_min

# each model should implement the get params function
def cv_multi_processes(model, data, params, folds_params):
    for param in params:
        for fold in folds(data, folds_params):
            m = model.fit(fold)
            error_in_fold = sum(abs(residuals(m)))
            yield param, error_in_fold

# class x_validator(object):
# two possibilities:
#  1. x-validation as something that creates different models
#  2. x-validation as a subroutine of the Model class.



# the above can be simply plugged to a pool of workers
# it can be an iterator
# it can be member of the class
