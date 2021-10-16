import numpy as np


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

    return loss, loss_prob, loss_sum, loss_exp


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

        # set linear rate for optimizing
        self.lr = lr

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

        for layer in self.architect:

            # get layer's input and output number

            in_num = layer["input_num"]
            out_num = layer["output_num"]
            batch_norm_status = layer["batch-norm"]

            # create theta randomly
            layer_theta = np.random.randn((in_num, out_num))

            # create theta's gradiant
            layer_theta_grad = np.zeros((in_num, out_num))

            # initialize norm_out
            norm_out = np.zeros(out_num)

            # create batch_norm layer for layer
            if batch_norm_status:

                # initialize mean and std for layer
                norm_mean = np.random.randn((1, out_num))
                norm_std = np.random.randn((1, out_num))

                # add to Batch_Norm_Theta list
                self.Batch_Norm_Theta.append([True, norm_mean, norm_std])

            else:

                # determine we didn't normalize output
                self.Batch_Norm_Theta.append([False])

            # output gradient
            output_grad = np.zeros(out_num)

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
                input_data, x_mean = normalize_input_data(input_data, norm_param[1], norm_param[2])

                # store Norm_Out 's index
                self.Norm_Out[layer_num] = x_mean

            # get activation fun name
            activation_name = self.architect[layer_num]["activation"]
            if activation_name == "relu":
                # relu activation
                input_data = self.relu(input_data)

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
            loss, self.output_probability, self.sum_of_exp, self.exp_out = cross_loss(prediction_data, label_data, self.size_avg)

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
        grad = self.cost_fun_grad(label)

        # calculate layer's gradient
        for layer_index in range(len(self.architect)-1, -1, -1):

            # layer
            layer = self.architect[layer_index]



    def cost_fun_grad(self, label):

        """
        calculate the cost function gradient

        :param label:       output's Truth label
        :return:            gradient
        """

        # last layer's Gradient
        last_layer_theta = self.Theta[-1]
        num_out_layer = last_layer_theta.shape[1]

        # initialize grad in the last layer
        grad = np.ones(num_out_layer)

        # if size_svg in loss function == True then we must divide it to num_out else we didn't do anything
        if self.size_avg:
            grad = grad / num_out_layer

        # grad = grad * Yi
        grad = grad * label

        # log derivation
        grad = grad * (1 / (np.log(10) * self.output_probability))

        # exp derivation
        first_grad = (1 / self.sum_of_exp) * grad
        second_grad = -1 * self.exp_out * grad * (1 / self.sum_of_exp ** 2)

        grad = first_grad + second_grad

        # final cost function gradient
        # this is the input gradient for layer's gradient
        grad = np.sum(grad, axis=0)

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

            # get layer's theta and gradient
            theta = self.Theta[layer_num]
            grad = self.Theta_Grad[layer_num]

            # optimize theta
            new_theta = theta + (self.lr * grad * -1)

            # update theta
            self.Theta[layer_num] = new_theta

    #######################
    #######################
    """    fit model    """

    #######################
    #######################

    def fit(self):
        pass
