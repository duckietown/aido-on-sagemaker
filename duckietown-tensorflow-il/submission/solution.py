#!/usr/bin/env python

import gym
# noinspection PyUnresolvedReferences
import gym_duckietown_agent  # DO NOT CHANGE THIS IMPORT (the environments are defined here)
from duckietown_challenges import wrap_solution, ChallengeSolution, ChallengeInterfaceSolution, InvalidEnvironment

import numpy as np
expect_shape = (480, 640, 3)

def check_valid_observations(observations):
    assert isinstance(observations, np.ndarray), type(observations)
    if observations.shape != expect_shape:
        msg = 'I expected size %s, while I got size %s' % (expect_shape, observations.shape)
        raise InvalidEnvironment(msg)



def solve(params, cis):
    # python has dynamic typing, the line below can help IDEs with autocompletion
    assert isinstance(cis, ChallengeInterfaceSolution)
    # after this cis. will provide you with some autocompletion in some IDEs (e.g.: pycharm)
    cis.info('Creating model.')
    # you can have logging capabilties through the solution interface (cis).
    # the info you log can be retrieved from your submission files.

    # BEGIN SUBMISSION
    # if you have a model class with a predict function this are likely the only lines you will need to modifiy
    from model import TfInference
    # define observation and output shapes
    model = TfInference(observation_shape=(1,) + expect_shape,  # this is the shape of the image we get.
                        action_shape=(1, 2),  # we need to output v, omega.
                        graph_location='tf_models/')  # this is the folder where our models are stored.
    # END SUBMISSION
    try:

        # We get environment from the Evaluation Engine
        cis.info('Making environment')
        env = gym.make(params['env'])
        # Then we make sure we have a connection with the environment and it is ready to go
        cis.info('Reset environment')
        observation = env.reset()
        check_valid_observations(observation)

        cis.info('Obtained first observations.')
        # While there are no signal of completion (simulation done)
        # we run the predictions for a number of episodes, don't worry, we have the control on this part
        while True:
            # we passe the observation to our model, and we get an action in return
            action = model.predict(observation)
            # we tell the environment to perform this action and we get some info back in OpenAI Gym style
            observation, reward, done, info = env.step(action)
            check_valid_observations(observation)
            # here you may want to compute some stats, like how much reward are you getting
            # notice, this reward may no be associated with the challenge score.

            # it is important to check for this flag, the Evalution Engine will let us know when should we finish
            # if we are not careful with this the Evaluation Engine will kill our container and we will get no score
            # from this submission
            if 'simulation_done' in info:
                cis.info('Received simulation_done.')
                break
            if done:
                cis.info('End of episode')
                env.reset()

    finally:
        cis.info('Releasing CPU/GPU resources.')
        # release CPU/GPU resources, let's be friendly with other users that may need them
        model.close()

    cis.info("Graceful exit of solve().")



### Leave the following boilerplate code alone

class Submission(ChallengeSolution):
    def run(self, cis):
        assert isinstance(cis, ChallengeInterfaceSolution)

        # get the configuration parameters for this challenge
        params = cis.get_challenge_parameters()
        cis.info('Parameters: %s' % params)

        solve(params, cis)  # let's try to solve the challenge,

        cis.set_solution_output_dict({})
        cis.info('Finished.')


if __name__ == '__main__':
    print('Starting submission')
    wrap_solution(Submission())

### (end)
