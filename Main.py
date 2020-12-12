from scripts.PreProcessingPipeline import PreProcessingPipeline
from scripts.TrainingPipeline import TrainingPipeline
import logging
import sys
from datetime import datetime
import optuna

"""
Entry point for meningioma machine learning project
"""
if __name__ == '__main__':
    logging.basicConfig(filename='./logs/meningioma_' + str(datetime.now()) + '.log', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    """
    Let's preprocess the images!
    """
    # preprocessingPipeline = PreProcessingPipeline()

    # preprocessingPipeline.Run()

    """
    Let's train!
    """
    trainingPipeline = TrainingPipeline(targets=['0', '1', '2', '3'])

    study = optuna.create_study(direction="maximize")
    study.optimize(trainingPipeline.objective, n_trials=100)

    trial_ = study.best_trial
    print(trial_.values)
    print(trial_.params)

    trainingPipeline.Run(trial_.params)




