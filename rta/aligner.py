from rta.preprocessing.preprocessing import preprocess


class Aligner(object):
    def __init__(self, 
                 preprocessing_args={}):
        """Initialize by preprocessing data."""
        self.dp = preprocess(**preprocessing_args)

    def calibrate_parameters(self):
        pass

    def update_alignment(self):
        pass
