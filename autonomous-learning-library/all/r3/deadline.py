import math

class Deadline(object):

    def __init__(self, lambda_param=0.0, D=0.0, debug=False):

        # Parameters
        self.D = D
        self.lambda_param = lambda_param

        # Variables to keep track of the history
        self.running_episode_history_sum = 0
        self.running_episode_gamma_history_sum = 0

        # Debug variables
        self.debug = debug
        self.episode_history = []
        self.gamma_history = []

    def calculate_episode_deadline(self, episode_number=1, last_episode_status=False):
        """
        Function to calculate the deadline of the new episode.

        @param episode_number - The episode number for which the deadline needs to be calculated.
        @return - A floating point value identifying the deadline.
        """

        # Calculate current episode number gamma
        episode_gamma = self.deadline_gamma(episode_number, last_episode_status)

        # Calculate numerator
        numerator = math.exp(self.lambda_param * (episode_number - 1)) * episode_gamma

        # Calculate denominator
        denominator = 1
        if episode_number > 1:
            denominator = self.running_episode_history_sum

        # Add to the running sum
        self.running_episode_history_sum += numerator

        if self.debug:

            # Append history
            self.episode_history.append(numerator)
            self.gamma_history.append(last_episode_status)

        # Return the deadline value
        return self.D * (numerator / denominator)

    def deadline_gamma(self, episode_number=1, last_episode_status=False):
        """
        Function to calculate the gamma value for one episode.

        @param episode_number - The episode number for which the gamma value needs to be calculated.
        @return - A floating point value realizing the gamma value.
        """

        # Per the equation, if it's the first episode
        if episode_number == 1:
            return 1

        # Calculate numerator for the gamma function
        self.running_episode_gamma_history_sum += 0
        if last_episode_status:
            self.running_episode_gamma_history_sum += 1

        return (self.running_episode_gamma_history_sum / (episode_number - 1))

    def reset(self, lambda_param=0.0, D=0.0, debug=False):
        """
        Function to reset all the parameters for the equation.

        @param lambda_param - The lambda value that needs to be reset.
        @param D - The universal D paramater that needs to be reset.
        @param debug - The debug flag
        """

        # Reset parameters
        self.lambda_param = lambda_param
        self.D = D
        self.debug = debug

        # Clear all items from the history
        self.gamma_history.clear()
        self.episode_history.clear()