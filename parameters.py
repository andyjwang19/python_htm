#  when picking activations chose the one less picked
#  low/high threshold for boost should be based on average number of activations * a ratio num
#  issue with boosting??

from libs import *

seed = random.randint(0,100)
random.seed(seed)
np.random.seed(seed)
#  probably don't need half of these libraries...

# testing purposes only
test_coord = []
# HYPERPARAMETERS

# for initalizing random input vectors
num_of_vectors = 2
vector_output_grid = 1 # math.ceil(num_of_vectors **0.5)
vector_len = 50

encode = True


# number of feed-forward connections from the input to each column
number_of_connections = 15
# how much the feed-forward weights change by each time step
upward_weight_change = 0.03
downward_weight_change = 0.05
# size of layer containing the columns
net_width = 32
net_height = net_width
# keep at zero because radius of inhibition is not a feature that should be implemented with the current setup
radius_of_inhibition = 0
# is epochs the right word?
num_of_epochs = 5000
# maximum amount of active columns
total_active = 20
# container for all the active columns
active_columns = []

# Temporal pooler stuff ignore for now
number_of_distal = 200
total_cells = number_of_distal
upward_distal_weight_increment = 0.005
downward_distal_weight_increment = 0.003
boost_amount = 0

testing_tp = True
# test for error
result = [[], []]


check_to_boost = 1000000
boost_step = 0  # 0.06
boost_low_threshold = 0.1 * check_to_boost
boost_high_threshold = 0.9 * check_to_boost
max_boost = 0.3
min_boost = -0.3

# how much the test vector differs from the input vector for each input vector
change_factor = 1

run_training = True
temporal_p = True

temp_boost = False

testing_adjacent = False


letter_encode = True
repeated_word = "abca"
