import os

# Validate config attributes read from a DAT file.


class ValidateConfig(object):
    @staticmethod
    def validate(data):
        # Validate dataset
        dataset = data.dataset
        if(dataset < 0):
            raise Exception('Wrong dataset configuration')

        # Validate groupDetection
        groupDetection = data.groupDetection
        if(dataset < 0):
            raise Exception('Wrong group detection configuration')

        # Validate groupModelling
        groupModelling = data.groupModelling
        if(dataset < 0):
            raise Exception('Wrong group modelling configuration')

        # Validate solution file
        solutionFile = data.solutionFile
        if(len(solutionFile) == 0):
            raise Exception('Value for solutionFile is empty')
