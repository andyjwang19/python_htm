from __future__ import division
from libs import *
from parameters import *


class column(object):
    def __init__(self, x_value, y_value, index_vector):
        self.x = x_value
        self.y = y_value

        self.value = 0
        self.connections = []
        self.weights = []

        self.distal = []
        self.distal_weights=[]

        # Each cell contains the coordinates of the column that sent the distal signal
        self.cells = [{"Predicted": False, "When": -1, "From": []} for x in range(0, total_cells)]

        self.times_fired = 0
        self.boost = 0


        # initialize lateral connections. for temporal pooler, ignore
       # for c in range(0,number_of_distal):
       #     self.distal.append([random.randint(0,net_width-1), random.randint(0,net_height-1)])
       #     # Not normal distribution... genconnnections is...
       #     self.distal_weights.append(0.5)
        # print(self.distal)
        # print(self.distal_weights)


        # initialize feed forward connections
        self.connections = np.random.choice(index_vector, number_of_connections, False)
        for i in range(0, number_of_connections):
            # distributes the connections with equal amounts above and below the threshold
            if i % 2 == 0:
                self.weights.append(np.random.uniform(0.1,1))
            else:

                self.weights.append(np.random.uniform(0,0.1))

    def generatePredictions(self, distal_net):
        for q in range(0, number_of_distal):
            x = random.randint(0,len(distal_net) - 1)
            # # print(x)
            y = random.randint(0,len(distal_net[x])-1)
            # print(y)

            potential = distal_net[x][y]
            for i in range(0,len(potential.cells)):
                if potential.cells[i]["From"] == []:
                    potential.cells[i]["From"] = [self.x,self.y]
                    self.distal.append([potential.x, potential.y])
                    if len(self.distal) % 2 == 0:
                        self.distal_weights.append(0.11)
                    else:
                        self.distal_weights.append(0.11)
                    if i == len(potential.cells) - 1:
                        del distal_net[x][y]
                        if distal_net[x] == []:
                            del distal_net[x]
                    break
            else:
                print("Something went wrong when generating the predictions")
        return distal_net


    def calculateValue(self, input_vector):
        # reset value everytime so value is accurately returned everytime it is called
        self.value = 0
        for i in range(0,len(self.connections)):
            if self.weights[i] > 0.1:
                # code commented out at the end of the next line is for weighted sum. Weighted sum is not part of the Numenta implementations and so is not part of our implmentation
                self.value += input_vector[self.connections[i]] # * self.weights[i]

        self.value += self.value * self.boost


        # disgusting test code but still probabaly necessary when code breaks

       # print(str(self.x),str(self.y))
       # print(self.value)
       # if self.x == 0 and self.y == 0:
       #     print(str(self.x) +', ' + str(self.y) + ' value: ' + str(self.value))
       #     temp = 0
       #     for i in range(0,len(self.connections)):
       #         temp += input_vector[self.connections[i]]
       #     print(str(self.x) +', ' + str(self.y) + ' Potential value: ' + str(temp))
       # if test_coord != []:
           # test_coord_2 = [1,1]
           # temp = False
           # while not temp:
           #     temp = True
           #     test_coord_2[0] += 1
           #     for i in range(0,len(test_coord)):
           #         if test_coord_2 == test_coord[i]:
           #             temp = False

       # if test_coord !=[] and self.x == test_coord[0] and self.y == test_coord[1]:
       #     print(self.weights)
       #     for i in range(0,len(self.connections)):
       #         print(input_vector[self.connections[i]])
               # print("value: " + str(self.value))
               # temp = 0
               # for i in range(0,len(self.connections)):
               #     temp += input_vector[self.connections[i]]
               # print("potential Value: " + str(temp))
        return self.value

#     def genDistal(self):
    # for Temporal Pooler
   # def activateCells(self,x,y):
   #     for b in range(0, len(self.cells)):
   #         if self.cells[b] == []:
   #             self.cells[b] =[x,y]


   # def distalSignal(self):
   #    for f in range(0,len(self.distal)):
   #         if self.distal_weight[f] > 0.1:
   #             net[self.distal[f]].activateCells(self.x,self.y)


   # def updateDistal(self, x, y):
   #     for f in range(0, len(self.distal)):
   #         if self.distal[f] == [x,y]:
   #             self.distal_weights[f] += weight_change
    # not for temporal pooler
    # Columns that are active go here. Are "fired", like a neuron. Updates weights connected to input and sends predictions to lateral connections + updates weights if it itself was predicted
    def fire(self, net, input_vector, temp, epoch):

        # more disgusting test code

        global test_coord
       # global active_columns
       # active_columns.append([self.x,self.y])
        if test_coord == []:
            test_coord = [self.x,self.y]
       # if self.x == test_coord[0] and self.y == test_coord[1]:
       #     print(str(self.x) +', ' + str(self.y))
       #     print(self.weights)
       #     for i in range(0,len(self.connections)):
       #         print(input_vector[self.connections[i]]),

        # update weights for lateral connects
        # cells contain the coordinates of sending column if predicted
       # for cell in self.cells:
       #     if cell == []:
       #         break
       #     # print(self.x,self.y)
       #     net[cell[0]][cell[1]].updateDistal(self.x,self.y)

        # print(str(self.x), str(self.y))


        # if connections are connected to a 1, it goes up, even if the weight was below threshold. If it was connected to a zero, then weight goes down no matter what because it doesn't want that kind of negative energy in its life
        small_input = [input_vector[self.connections[x]] for x in range(0, len(self.connections))]
        for i in range(0,len(small_input)):
            if small_input[i] == 0:
                if self.weights[i]- downward_weight_change > 0:
                    self.weights[i] -= downward_weight_change
                else:
                    self.weights[i] = 0
            if small_input[i] == 1:
                if self.weights[i] + upward_weight_change < 1:
                    self.weights[i] += upward_weight_change
                else:
                    self.weights[i] = 1



        # Update temporal pooler

        # send distal signals
        for i in range(0,len(self.distal)):
            if self.distal_weights[i] > 0.1:
                activate = self.distal[i]
                for q in range(0, len(net[activate[0]][activate[1]].cells)):
                    if net[activate[0]][activate[1]].cells[q]['From'] == [self.x, self.y]:
                        net[activate[0]][activate[1]].cells[q]['Predicted'] = True
                        net[activate[0]][activate[1]].cells[q]['When'] = epoch

        self.times_fired += 1
        # if self.x == 4 and self.y == 28:
        #     print(epoch)
        #     print(self.distal)
        #     print(self.distal_weights)



        # if temp == 3:
        #     print("Column: "+ str(self.x) +',' + str(self.y) + " epoch: " + str(epoch) +  " weights: ")
        #     print(self.weights)
        #     print(small_input)

         # send distal signals
        # for f in range(0,len(self.distal)):
        #     if self.distal_weights[f] > 0.1:
        #        # print(str(len(net)), str(len(net[0])))
        #        # print(self.distal[f][0])
        #        # print(self.distal[f][1])
        #         net[self.distal[f][0]][self.distal[f][1]].activateCells(self.x,self.y)



# generates and SDR given a net where each element is the value of the column. But does so pretty poorly
def gen_sdr(value_net):
    # problem if valuenet is only 0s

    sdr = []
    flat = sum(value_net, [])
    count = 0
    while max(flat) != 0 and count < total_active:
        flat = sum(value_net, [])
        max_num = np.argmax(flat)
        y = int(max_num%net_width)
        x = int(math.floor(max_num/net_width))

        value_net[x][y] = 0

        # radius of inhibition stuff that should NOT BE HERE
       # for q in range(x-radius_of_inhibition, x + 1+ radius_of_inhibition):
       #     for t in range(y-radius_of_inhibition, y + 1+ radius_of_inhibition):
       #         if q >= 0 and t >= 0 and q < net_height and t < net_width:
       #             value_net[q][t] = 0
        sdr.append((x,y))
        count += 1
    # this is the SDR but i didn't want to copy the value_net jsut so it could be renamed. when this function gets called the "value_net" will be changed into the sdr and then stored under a new variable
    # also the sdr is in the form of a 2D array that is mostly zeroes. the largest elements are stored as -1 for some reason that i forget.
    return sdr



def compareVectors(vec1, vec2):
    count = 0
    for element1 in vec1:
        for element2 in vec2:
            if element1 == element2:
                count += 1
                break
    return count, len(vec1)


# main bit of function. This creates the value_net, which is a grid of all the "scores" of all the columns, then turns that value_net into an SDR. Finally, it loops through the SDR and finds the sparse bits and fires them.
# also plot is an int that i treat as a bool bc i am lazy
def step(net, input_vector, index_vector, epoch, train, plot, old_sdr, temporal_train=True, shape=None, old_plot=False, old_shape='ro',
         plot_name='okok', old_name='oldold', current_prediction=False, cp_shape='g^', cp_name='cu prediction'):
    # prints current epoch
    if not plot and train:
        print(str(epoch+1) + '/' + str(num_of_epochs))

    value_net = []
    iteration = 0
    for ar in net:
        value_net.append([])
        for obj in ar:
        #    if epoch == 0:
        #        obj.generateConnections(index_vector)
        #        obj.genDistal()
            value_net[iteration].append(obj.calculateValue(input_vector))
        iteration+= 1
    # print(value_net)

    # creates sparse distributed representation
    sdr = gen_sdr(value_net)
    sdr_net_copy = copy.deepcopy(sdr)
    # print(sdr_net)

    # trains if test is off, that is, if you aren't testing
    if train:
        temp = 0
        for x in range(0, len(sdr)):
               net[sdr[x][0]][sdr[x][1]].fire(net, input_vector, temp, epoch)
               temp += 1

        # way messy clean up later
        if temporal_train:
            if old_sdr != False:
                for i in range(0, len(old_sdr)):
                    for q in range(0, len(net[sdr[i][0]][sdr[i][1]].distal)):
                        # if old sdr columns predict current active columns
                        if (net[ old_sdr[i][0] ][ old_sdr[i][1] ].distal[q][0], net[old_sdr[i][0]][old_sdr[i][1]].distal[q][1]) in sdr:
                            # upward increment unless it is at ceiling in which case set to 1
                            if net[old_sdr[i][0]][old_sdr[i][1]].distal_weights[q] + upward_distal_weight_increment < 1:
                                net[old_sdr[i][0]][old_sdr[i][1]].distal_weights[q] += upward_distal_weight_increment
                            else:
                                net[old_sdr[i][0]][old_sdr[i][1]].distal_weights[q] = 1
                        else:
                            if net[old_sdr[i][0]][old_sdr[i][1]].distal_weights[q] - downward_distal_weight_increment > 0:
                                net[old_sdr[i][0]][old_sdr[i][1]].distal_weights[q] -= downward_distal_weight_increment
                            else:
                                net[old_sdr[i][0]][old_sdr[i][1]].distal_weights[q] = 0
                if temp_boost:
                    for x in range(0, len(sdr)):
                        for q in range(0, len(net[x][y].cells)):
                            coor = net[sdr[x][0]][sdr[x][1]].cells[q]["From"]
                            if coor != []:
                                for r in range(0, len(net[coor[0]][coor[1]].distal)):
                                    if net[coor[0]][coor[1]].distal[r] == [x, y]:
                                        net[coor[0]][coor[1]].distal_weights[r] += boost_amount

    # plots
    if plot:
        ax = plt.subplot(4, 2, plot)
        ax.axis([0, 32, 0, 32])

        ax.set_title("Vector: " + plot_name)
        for x in range(0, len(sdr)):
            ax.plot([sdr[x][0]], [sdr[x][1]], shape)

    if old_plot:
        ax = plt.subplot(4, 2, old_plot)
        ax.axis([0, 32, 0, 32])
        print(old_sdr)
        ax.set_title("Prediction: " + old_name)
        for x in range(0, len(old_sdr)):
            col = net[old_sdr[x][0]][old_sdr[x][1]]
            for t in range(0,len(col.distal)):
                if col.distal_weights[t] > 0.1:
                   # print(col.distal[t])
                    ax.plot([col.distal[t][0]], [col.distal[t][1]] , old_shape)
    if current_prediction:
        predicted_sdr = []
        ax = plt.subplot(4, 2, current_prediction)
        ax.set_title("Prediction from: " + cp_name)
        for x in range(0, len(sdr)):
            col = net[sdr[x][0]][sdr[x][1]]
            for t in range(0,len(col.distal)):
                if col.distal_weights[t] > 0.1:
                    # print(col.distal[t])
                    predicted_sdr.append((col.distal[t][0], col.distal[t][1]))
                    ax.plot([col.distal[t][0]], [col.distal[t][1]], cp_shape)

    if current_prediction:
        print(predicted_sdr)
        print("\n")
        print(sdr_net_copy)
        return net, sdr_net_copy, predicted_sdr
    else:
        return net, sdr_net_copy


def boost(net):
    print("BOOSTED")
    counter = 0


   # testx = 0
   # testy = 7
    for x in range(0, len(net)):
        for y, column in enumerate(net[x]):
           # if x == testx and y == testy:
           #     print(net[x][y].times_fired)
           #     print(net[x][y].boost)
            if column.times_fired < boost_low_threshold:
                # if boost % - step does not make boost % go over a certain amount
                if net[x][y].boost + boost_step <= max_boost:
                    net[x][y].boost += boost_step
                    counter += 1
                else:
                    net[x][y].boost = max_boost
                    counter += 1
                   # if x == testx and y == testy:
                   #     print("up")
                # add to the boost %
            elif column.times_fired > boost_high_threshold:
                print(net[x][y].boost)
                if net[x][y].boost - boost_step >= min_boost:
                    net[x][y].boost -= boost_step
                    counter += 1
                else:
                    net[x][y].boost = min_boost
                    counter += 1
                   # if x == testx and y == testy:
                   #     print("down")
                # subtract from boost %

           # if x == testx and y == testy:
           #     print(net[x][y].boost)

            # reset
            net[x][y].times_fired = 0
    return net


# randomly generates vector of a given length. each element is either a 1 or a 0
def generateVector(length):
    result = []
    for i in range(0, length):
        result.append(int(random.getrandbits(1)))
    return result


def sort(array):
    array=sorted(array, key=lambda x:(x[0], x[1]))
    return array


def encoder():
    N = 10
    W = 7
    result = []
    for position in range(0, N - W + 1):
        result.append([0 for x in range(0, position)] + [1 for x in range(0, W)] + [0 for x in range(W + position, N)])

    for position in range(N - W - 1, -1, -1):
        result.append([0 for x in range(0, position)] + [1 for x in range(0, W)] + [0 for x in range(W + position, N)])

    result = np.array(result)
    print(len(result))
    return result, N


def returnUnique(array):
    if array == []:
        return 0

    a = []
    for x in range(0, len(array)):
        for tup in array[x]:
            a.append(tup)
    return set(a)


def countUnique(array):
    te = returnUnique(array)
    if te != 0:
        return len(returnUnique(array))
    else:
        return 0


def main():

    global result
    # this parser is so unecessary and ineffective
    parser = argparse.ArgumentParser(description = "runs spatial pooling portion of htm model")
    parser.add_argument("--a", default = False)
    args = parser.parse_args()

    # inputs and test vectors are stored in arrays.
    if letter_encode:
        letter_vectors = []
        input_vector = []
        letters = ['a', 'b', 'c']
        for i in range(0, len(letters)):
            letter_vectors.append(generateVector(50))
        for i, let in enumerate(repeated_word):
            pos = letters.index(let)
            input_vector.append(letter_vectors[pos])
        print(letters)
        index_vector = [i for i in range(0, len(input_vector[0]))]


    else:
        input_vector = []
        test_vector = []
        if not encode:
            for w in range(0,num_of_vectors):
                input_vector.append(generateVector(vector_len))
                test_vector.append(copy.copy(input_vector[w]))
                # change a piece of the input vector
                # simulates a noisy but otherwise identical input
                for i in range(0, change_factor):
                    test_vector[w][i] = 1 - test_vector[w][i]
            # saves every index in input_vector. This is for the columns to randomly pick an index as their feed-forward input. Also so i can use random.choice and everything is easier
            index_vector = [i for i in range(0,vector_len)]
        else:
            input_vector, length = encoder()
            for w in range(0, len(input_vector)):
                test_vector.append(copy.copy(input_vector[w]))
                # change a piece of the input vector
                # simulates a noisy but otherwise identical input
                for i in range(0, change_factor):
                    test_vector[w][i] = 1 - test_vector[w][i]

            index_vector = [i for i in range(0,length)]
            # print(input_vector)

   # input_vector = input_vector[10:18]
   # length = len(input_vector)


    # print(input_vector[0])


    # 2D array that contains all the columns
    net = [[column(i, q, index_vector) for i in range(0,net_width)] for q in range(0,net_height)]
    distal_net=copy.deepcopy(net)
    for r in range(0,len(net)):
        for c in range(0,len(net[r])):
            distal_net = net[r][c].generatePredictions(distal_net)
    # print(net)





    # actual training, weights are updated and such
    sdr_net = False
    save = copy.copy(net[2][6].distal_weights)
    total_active = []
    test_1 = False
    if run_training:
        for epoch in range(0, num_of_epochs):
            if letter_encode:
               #if epoch % len(input_vector) == 0:
               #    net, sdr_net = step(net, input_vector[epoch % len(input_vector)], index_vector, epoch, True, 0,
               #                       sdr_net, temporal_train=False)
               #else:
                net, sdr_net = step(net, input_vector[epoch % len(input_vector)], index_vector, epoch, True, 0,
                                       sdr_net, temporal_train=True)


            else:
                # for i in range(0, num_of_vectors):
                # def step(net, input_vector, index_vector, epoch, test, plot, old_sdr):
                if testing_tp:
                    # net, sdr_net = step(net, input_vector[epoch % num_of_vectors], index_vector, epoch, True if epoch %
                    #                     num_of_vectors == 0 else False, 0, sdr_net)
                    net, sdr_net = step(net, input_vector[epoch % len(input_vector)], index_vector, epoch, True, 0, sdr_net)
                else:
                    net, sdr_net = step(net, input_vector[epoch % len(input_vector)], index_vector, epoch, True, 0, sdr_net)

                if epoch % len(input_vector)== 0:
                    if test_1:
                        print(compareVectors(returnUnique(total_active), test_1)[0])
                    print(countUnique(total_active))
                    test_1 = returnUnique(total_active)
                    total_active = []
                else:
                    total_active.append(sdr_net)


            if epoch % check_to_boost == 0:
                print('boosted')
                boost(net)
    else:
        epoch = 0
    for x in range(0, len(net)):
        temp = []
        for y, col in enumerate(net[x]):
            temp.append(round(col.boost, 4))


    if letter_encode:
        sdr_let = [0]*10
        sdr_predicted = [0]*10
        for x in range(0, len(letters)):
            print("let")
            print(x)
            current_letter = str(letters[x])
            net, sdr_let[x], sdr_predicted[x] = step(net, letter_vectors[x], index_vector, epoch, False, x+1, sdr_net, shape='ro',
                                   plot_name=current_letter, current_prediction=x+1+len(letters), cp_shape='b^',
                                   cp_name=current_letter)
        for x in range(0, len(letter_vectors)):
            for y in range(0, len(letter_vectors)):
                count, total = compareVectors(sdr_predicted[x], sdr_let[y])
                print(str(letters[x]) + " to " + str(letters[y]) + ": " + str(count) + "/" + str(total))
    elif testing_adjacent:
        avg = 0
        for x in range(0, len(input_vector)-1):
            net, first_sdr = step(net,input_vector[x], index_vector, epoch, False, x, sdr_net, shape='ro', plot_name=str(x))
            net, sec_sdr = step(net,input_vector[x+1], index_vector, epoch, False, x, sdr_net, shape='g^', plot_name=str(x))
            count, total = compareVectors(first_sdr, sec_sdr)
            print(str(x) + " and " + str(x+1) + ": " + str(count) + "/" + str(total))
            avg += count
        avg = avg/(len(input_vector)-1)
        print(avg)
    elif testing_tp and encode:

        net, sdr_net = step(net, input_vector[0], index_vector, epoch, False, False, sdr_net)
        net, sdr_net = step(net, input_vector[0], index_vector, epoch, False, 1, sdr_net, shape='ro', old_plot=5,
                            old_shape='b^', plot_name='input v 1', old_name='input v 1 prediction')

        net, sdr_net = step(net, input_vector[1], index_vector, epoch, True, False, sdr_net)
        net, sdr_net = step(net, input_vector[1], index_vector, epoch, False, 2, sdr_net, shape='ro', old_plot=6,
                            old_shape='b^', plot_name='input v 2', old_name='input v 2 prediction')

        net, sdr_net = step(net, input_vector[2], index_vector, epoch, True, False, sdr_net, True)
        net, sdr_net = step(net, input_vector[2], index_vector, epoch, False, 3, sdr_net, shape='ro', old_plot=7,
                            old_shape='b^', plot_name='input v 3', old_name='input v 3 prediction')

        net, sdr_net = step(net, input_vector[3], index_vector, epoch, True, False, sdr_net, True)
        net, sdr_net = step(net, input_vector[3], index_vector, epoch, False, 4, sdr_net, shape='ro', old_plot=8,
                            old_shape='b^', plot_name='input v 4', old_name='input v 4 prediction')
       #net, sdr_net = step(net, input_vector[0], index_vector, epoch, False, 0, sdr_net)
       #net, sdr_net = step(net, input_vector[1], index_vector, epoch, True, 1, sdr_net)
       #net, sdr_net = step(net, input_vector[2], index_vector, epoch, True, 2, sdr_net)
       #net, sdr_net = step(net, input_vector[3], index_vector, epoch, True, 3, sdr_net)

    elif testing_tp:
        net, sdr_net = step(net, input_vector[0], index_vector, epoch, False, False, sdr_net, current_prediction=1, cp_name=str(0), cp_shape='b^')
        temp = copy.deepcopy(sdr_net)
        net, sdr_net = step(net, input_vector[1], index_vector, epoch, False, 1, sdr_net, shape='ro')

       # print(save)
       # print(net[2][6].distal_weights)

       #net, sdr_net = step(net, input_vector[1], index_vector, epoch, False, 0, sdr_net)
       #net, sdr_net = step(net, input_vector[0], index_vector, epoch, True, 2, sdr_net)

    elif encode:
        print("ENDCODE")
        net, first_sdr = step(net,input_vector[0], index_vector, epoch, True, 1, sdr_net)
        net, sec_sdr = step(net,input_vector[1], index_vector, epoch, True, 1, sdr_net, True)
        count, total = compareVectors(first_sdr, sec_sdr)
        print("1st: " + str(count) + "/" + str(total))

        net, first_sdr = step(net,input_vector[0], index_vector, epoch, True, 2, sdr_net)
        net, sec_sdr = step(net,input_vector[2], index_vector, epoch, True, 2, sdr_net, True)
        count, total = compareVectors(first_sdr, sec_sdr)
        print("2nd: " + str(count) + "/" + str(total))

        net, first_sdr = step(net,input_vector[0], index_vector, epoch, True, 3, sdr_net)
        net, sec_sdr = step(net,input_vector[3], index_vector, epoch, True, 3, sdr_net, True)
        count, total = compareVectors(first_sdr, sec_sdr)
        print("3rd: " + str(count) + "/" + str(total))

        num = 4
        net, first_sdr = step(net,input_vector[0], index_vector, epoch, True, num, sdr_net)
        net, sec_sdr = step(net,input_vector[num], index_vector, epoch, True, num, sdr_net, True)
        count, total = compareVectors(first_sdr, sec_sdr)
        print(str(num) + "th: " + str(count) + "/" + str(total))
        num = 5
        net, first_sdr = step(net,input_vector[0], index_vector, epoch, True, num, sdr_net)
        net, sec_sdr = step(net,input_vector[num], index_vector, epoch, True, num, sdr_net, True)
        count, total = compareVectors(first_sdr, sec_sdr)
        print(str(num) + "th: " + str(count) + "/" + str(total))
        num = 6
        net, first_sdr = step(net,input_vector[0], index_vector, epoch, True, num, sdr_net)
        net, sec_sdr = step(net,input_vector[num], index_vector, epoch, True, num, sdr_net, True)
        count, total = compareVectors(first_sdr, sec_sdr)
        print(str(num) + "th: " + str(count) + "/" + str(total))
        num = 7
        net, first_sdr = step(net,input_vector[0], index_vector, epoch, True, num, sdr_net)
        net, sec_sdr = step(net,input_vector[num], index_vector, epoch, True, num, sdr_net, True)
        count, total = compareVectors(first_sdr, sec_sdr)
        print(str(num) + "th: " + str(count) + "/" + str(total))
    else:

        temp = []
        for i in range(0, len(input_vector)):
            result = [[],[]]

            net, first_sdr = step(net, input_vector[i], index_vector, epoch, False, i+1, False)


            if epoch == 0:
                epoch+= 1

            net, sec_sdr = step(net, test_vector[i], index_vector, epoch, True, i+1, False, True)
            count, total = compareVectors(first_sdr, sec_sdr)
            print(str(i + 1) + ": " + str(count) + "/" + str(total))
        count, total = compareVectors(first_sdr, sec_sdr)
        print(str(count) +"/" + str(total))
   # print(str(count) + "/" + str(total))
   # for i in range(0, num_of_vectors):

   #     step(net,input_vector[i],index_vector, epoch, False, i+1)
   #     if epoch == 0:
   #         epoch+= 1
  #  for i in range(0, num_of_vectors):

  #      step(net,input_vector[i],index_vector, epoch, False, i+1)
  #      if epoch == 0:
  #          epoch+= 1
  #      # step(net, test_vector[i], index_vector, epoch, True, i+1)

    error = 0
    # there is probably a better way to do this
    # reruns the step function once for each input_vector with the training off but the plotting on
    # also does the same for all the test_vectors
    # tries to save the errors but that doesn't work too well
    """
    for i in range(0, num_of_vectors):
        result = [[],[]]

        step(net,input_vector[i],index_vector, epoch, False, i+1, False)

        if epoch == 0:
            epoch+= 1

        step(net, test_vector[i], index_vector, epoch, True, i+1, False)

        result[0] = sort(result[0])
        result[1] = sort(result[1])
        print(result)
        for i in range(0, len(result[0])):
            if result[0][i] != result[1][i]:
                print(result)
                error += 1
                """
    print('ok')
    # print(error)
    plt.show()


if __name__ == "__main__":
    main()
