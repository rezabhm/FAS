import random
import sys
import numpy as np
import pickle


def relu(data):

    """
    get data and add relu activation

    :param data:            data that we must add relu fun
    """

    # find all of data that is under 0
    data_finder = np.where(data <= 0.0)

    # convert from numpy array to list
    rows = data_finder[0].tolist()
    columns = data_finder[1].tolist()

    data[rows, columns] = 0.0

    return data, data_finder


def normalize_input_data(x, mean, std):
    """
    normalize input data with mean and std based of this formula ==> ( x - mean) / std

    x:           ==> input data
    mean:        ==> mean param
    std:         ==> std param
    return:      ==> normalized data , (x - mean)
    """

    # calculate prediction - mean
    x_mean = x - mean

    # calculate (prediction - mean) / std
    x_final = x_mean / std

    return x_final, x_mean


def cross_loss(prediction_data, label_data, size_avg):
    # calculate loss = - sigma( log (p(y)) )

    # calculate prediction_data's exp
    loss_exp = np.exp(prediction_data)

    # sum loss_exp row's
    loss_sum = np.sum(loss_exp, axis=1)

    # calculate every data's probability
    loss_prob = loss_exp / loss_sum.reshape((loss_sum.shape[0], 1))

    # split label's probability
    label_probability = loss_prob[:, label_data]

    # calculate loss's log
    loss = np.log10(label_probability)

    # number of data
    m = loss.shape[0]

    # calculate final loss
    loss = loss.sum() * -1

    # calculate loss average
    if size_avg:
        loss = loss / m

    return loss, loss_prob, loss_sum.reshape((prediction_data.shape[0], 1)), loss_exp


class NN:

    """
    this is neural network class
    """

    def __init__(self, architect, loss="cross", size_avg=False, lr=1e-3):

        # neural network's architect
        self.architect = architect  # ===> this is list that every index is layer and every index is dict with blew key

        """
         key :

            ( name : layer name  , default : unknown )
            ( input_num : number of layer's input data )
            ( output_num : number of layer's output data)
            ( activation : choose one of this (relu , leaky_relu , tanh , sigmoid , False) , default is relu)
            ( batch-norm :  determine use batch normalization or not .
                this is boolean var (True , False) , default False  )
            ( dropout : determine dropout amount (0-100) , if set 0 it equal didn't use dropout , default is 0)

        """
        # forward param
        # store every layer's output
        self.layers_output = []

        # param for initialize model
        # theta param
        self.Theta = []

        # theta's gradient
        self.Theta_Grad = []
        self.OutPut_Grad = []

        # output's probability through  the calculate loss fun
        self.output_probability = None
        self.sum_of_exp = None

        # batch normalization theta
        self.Batch_Norm_Theta = []

        # for store output - mean
        self.Norm_Out = []

        # initialize Theta
        self.init_theta()

        # set cost fun , default is cross
        self.cost_fun_name = loss
        # set size average status . if it's true we will calculate average loss .
        self.size_avg = size_avg

        # blew list contain all of layer's dropout index
        self.dropout_list = []

        # for store all of parameter that is under zero and we will change it to 0.
        self.activation_grad_need = []

        # add exp to output layer
        self.exp_out = None

        # set linear rate for optimizing
        self.lr = lr

        # models input data (train data)
        self.train_data = None

    def init_theta(self):

        """
        initialize Theta and theta's gradiant
        """

        self.Theta = []
        self.Theta_Grad = []
        self.OutPut_Grad = []
        self.Batch_Norm_Theta = []

        # for store output - mean
        self.Norm_Out = []

        # dropout list
        self.dropout_list = []

        for layer in self.architect:

            # get layer's input and output number

            in_num = layer["input_num"]
            out_num = layer["output_num"]
            batch_norm_status = layer["batch-norm"]
            dropout_amount = layer["dropout"]

            # create theta randomly
            layer_theta = np.random.randn((in_num, out_num))

            # create theta's gradiant
            layer_theta_grad = np.zeros((in_num, out_num))

            # initialize norm_out
            norm_out = np.zeros((1, out_num))

            # create batch_norm layer for layer
            if batch_norm_status:

                # initialize mean and std for layer
                norm_mean = np.random.randn((1, out_num))
                norm_mean_grad = np.random.randn((1, out_num))
                norm_std = np.random.randn((1, out_num))
                norm_std_grad = np.random.randn((1, out_num))

                # add to Batch_Norm_Theta list
                self.Batch_Norm_Theta.append([True, norm_mean, norm_std, norm_mean_grad, norm_std_grad])

            else:

                # determine we didn't normalize output
                self.Batch_Norm_Theta.append([False])

            # output gradient
            output_grad = np.zeros(out_num)

            # determine dropout index
            dropout_amount = dropout_amount/100
            dropout_amount_num = int(out_num * dropout_amount)

            # create dropout index randomly
            dropout_list_layer = []
            layer_range = list(range(out_num))
            for i in range(dropout_amount_num):

                randint = random.randint(0, len(layer_range)-1)
                val = layer_range[randint]
                _ = layer_range.remove(val)
                dropout_list_layer.append(val)

            self.dropout_list.append(dropout_list_layer)

            # add data
            self.Theta.append(layer_theta)
            self.Theta_Grad.append(layer_theta_grad)
            self.OutPut_Grad.append(output_grad)
            self.Norm_Out.append(norm_out)

    #######################
    #######################
    """    forward      """

    #######################
    #######################

    def forward(self, input_data):

        """
        forward through the model's layer

        input_data : our data , shape ==> (m,n) m : number of data , n : number feature's
        """

        # store every layer's output
        self.layers_output = []

        # for store all of parameter that is under zero and we will change it to zero
        self.activation_grad_need = []

        # store models input data (train data)
        self.train_data = input_data

        # calculate layer's output
        for layer_num in range(len(self.Theta)):

            # get layer's theta
            layer_theta = self.Theta[layer_num]

            # manipulate theta * data
            input_data = input_data @ layer_theta

            # Normalize output
            # get norm param
            norm_param = self.Batch_Norm_Theta[layer_num]
            if norm_param[0]:

                # normalized data with mean and std
                input_data, x_mean = normalize_input_data(input_data, norm_param[1], norm_param[3])

                # store Norm_Out 's index
                self.Norm_Out[layer_num] = x_mean

            # get activation fun name
            activation_name = self.architect[layer_num]["activation"]
            if activation_name == "relu":

                # relu activation
                input_data, activation_grad_param = relu(input_data)

            else:

                # this is a list with 2 index in neural network that first index is row and second is column
                activation_grad_param = []

            self.activation_grad_need.append(activation_grad_param)

            # add dropout
            dropout_list = self.dropout_list[layer_num]

            if len(dropout_list) > 0:

                input_data[:, dropout_list] = 0.0

            # add layer's output to output's list
            self.layers_output.append(input_data)

        return self.layers_output[-1]

    #######################
    #######################
    """    loss         """

    #######################
    #######################

    def cost_fun(self, prediction_data, label_data):

        """
        calculate model's loss

        prediction_data :   model's output(predict) data , shape ==> (m , o) m : number
                            of data , o : number of output's feature
        label_data :        data's truth label , shape == > (m , ) , m : number of data
        """

        label_data = label_data.tolist()
        loss = 0

        if self.cost_fun_name == "cross":
            # calculate cross entropy loss
            loss, self.output_probability, self.sum_of_exp, self.exp_out = cross_loss(prediction_data, label_data,
                                                                                      self.size_avg)

        return loss

    def zero_grad(self):

        """
        clear previous gradient

        :return:
        """

        for layer_num in range(len(self.Theta_Grad)):
            # get gradient
            grad = self.Theta_Grad[layer_num]

            # create zero gradient
            zero_grad = np.zeros(grad.shape)

            # update gradient
            self.Theta_Grad[layer_num] = zero_grad

    #######################
    #######################
    """    backward     """

    #######################
    #######################

    def backward(self, label):

        """
        calculate gradient with backward algorithm

        label :     output's Truth label ==> (m, number_of_output_layer)
        :return:
        """

        # calculate cost fun gradient
        # data is (m,n) shape
        grad = self.cost_fun_grad(label)

        # calculate layer's gradient
        for layer_index in range(len(self.architect)-1, -1, -1):

            # calculate dropout gradient
            dropout_list = self.dropout_list[layer_index]

            if len(dropout_list) > 0:

                # change grad param to zero
                grad[:, dropout_list] = 0.0

            # calculate activation gradient
            gradient_need = self.activation_grad_need[layer_index]

            if len(gradient_need) > 0:

                # get index from gradient_need param
                row = gradient_need[0].tolist()
                column = gradient_need[1].tolist()

                # change all of parameter to 0. because in forward step we had change to 0 so it's
                # backward param is 0.0
                grad[row, column] = 0.0

            # normalize gradient
            if self.architect[layer_index]["batch-norm"]:

                # get batch-norm data
                batch_norm_x = self.Batch_Norm_Theta[layer_index]

                # calculate std gradient
                batch_norm_x[-1] = grad * (
                                                (-1 * self.Norm_Out[layer_index])
                                                /
                                                (batch_norm_x[2]**2)
                                           )

                # calculate numerator gradient in normalizing
                grad = grad / batch_norm_x[2]

                # calculate mean theta gradient
                batch_norm_x[3] = -1 * grad / batch_norm_x[2]

                # update gradient
                self.Batch_Norm_Theta[layer_index] = batch_norm_x

            # Theta and input's gradient

            if layer_index != 0:

                # for all of layer's except of first layer because first's layer's input is out train data
                # we must multiplication with previous layer's output
                self.Theta_Grad[layer_index] = grad * self.layers_output[layer_index - 1]

            elif layer_index == 0:

                # multiplication with NN input data
                self.Theta_Grad[layer_index] = grad * self.train_data

            # calculate input's data gradient == Theta * grad
            # this layer's input's data's grad is previous layer's first grad
            grad = self.Theta[layer_index] * grad

    def cost_fun_grad(self, label):

        """
        calculate the cost function gradient

        :param label:       output's Truth label
        :return:            gradient
        """

        # last layer's Gradient
        last_layer_theta = self.Theta[-1]
        num_out_layer = last_layer_theta.shape

        # initialize grad in the last layer
        grad = np.ones(num_out_layer)

        # if size_svg in loss function == True then we must divide it to num_out else we didn't do anything
        if self.size_avg:
            grad = grad / num_out_layer[0]

        # grad = grad * Yi
        grad = grad * label

        # log derivation
        grad = grad * (1 / (np.log(10) * self.output_probability))

        # exp derivation
        first_grad = grad / self.sum_of_exp
        second_grad = -1 * self.exp_out * grad * (1 / (self.sum_of_exp ** 2))

        grad = first_grad + second_grad

        # final cost function gradient
        # this is the input gradient for layer's gradient
        # grad = np.sum(grad, axis=0)

        return grad

    #######################
    #######################
    """    optimize     """

    #######################
    #######################

    def optimize(self):

        """
        this fun will optimize Theta for training model.
        """

        # optimize every layer's theta in a different step
        for layer_num in range(len(self.Theta)):

            # optimize model's Theta
            # get layer's theta and gradient
            theta = self.Theta[layer_num]
            grad = self.Theta_Grad[layer_num]

            # optimize theta
            new_theta = theta + (self.lr * grad.sum(axis=0) * -1)

            # update theta
            self.Theta[layer_num] = new_theta

            # optimize normalize Theta
            if self.Batch_Norm_Theta[layer_num][0]:

                # get theta and gradient param
                mean_theta = self.Batch_Norm_Theta[layer_num][1]
                mean_theta_grd = self.Batch_Norm_Theta[layer_num][3]
                std_theta = self.Batch_Norm_Theta[layer_num][2]
                std_theta_grd = self.Batch_Norm_Theta[layer_num][4]

                # add linear rate
                mean_theta_grd = -1 * self.lr * mean_theta_grd.sum(axis=0)
                std_theta_grd = -1 * self.lr * std_theta_grd.sum(axis=1)

                # optimize
                mean_theta = mean_theta + mean_theta_grd
                std_theta = std_theta + std_theta_grd

                # update param
                self.Batch_Norm_Theta[layer_num][1] = mean_theta
                self.Batch_Norm_Theta[layer_num][2] = std_theta

    #######################
    #######################
    """    fit model    """
    #######################
    #######################

    def fit(self, train_data_x, train_data_y, epoch_num=10, batch_size=16):

        """
        start train model

        train_data_x :        this is array (m,n) that include all of data for training model
        train_data_y :        this is array (m, ) that include above data's label
        epoch_num    :        this is epoch number that determine number of epoch for training data
        batch_size   :        define number of data batch
        """

        best_loss = float('inf')

        # start training
        # epoch loop
        for epoch in range(epoch_num):

            avg_loss = 0.0
            num_data = 0

            # batch size loop
            for batch in range(0, train_data_x.shape[0], batch_size):

                # calculate range of data for training based of batch size
                batch_finish = batch + batch_size if batch+batch_size < train_data_x.shape[0] else train_data_x.shape[0]

                # get batch data
                batch_data_x = train_data_x[batch:batch_finish, :]
                batch_data_y = train_data_y[batch:batch_finish]

                # forward
                prediction_data = self.forward(batch_data_x)

                # zero grad
                self.zero_grad()

                # calculate data loss
                loss = self.cost_fun(prediction_data, batch_data_y)

                # backward
                self.backward(batch_data_y)

                # optimize Theta
                self.optimize()

                # calculate average loss
                avg_loss = (loss + (avg_loss * num_data)) / (num_data + (batch_finish - batch))
                num_data += batch_finish - batch

                # report result
                sys.stdout.flush()
                sys.stdout.write("\r [{0}/{1}]training loss : {2} && average loss : {3}".format(str(batch),
                                                                                                str(
                                                                                                train_data_x.shape[0]
                                                                                                ), str(loss), str(
                                                                                                            avg_loss)))

            if avg_loss <= best_loss:

                # store model's Theta
                with open("best_model.pickle", "wb") as fd:
                    pickle.dump([self.Theta, self.Batch_Norm_Theta, self.architect], fd)

                avg_loss_best = "Stored"

            else:
                avg_loss_best = "Don't Stored"

            # report epochs loss
            print("Epoch : {0} -- training loss == {1} -- {2}".format(str(epoch), str(avg_loss), avg_loss_best))
