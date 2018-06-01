import cPickle

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class SA(object):
    """ Long Short-Term Memory Sequence-to-Sequence Autoencoder
        (with peephole connections)

        References: [1] http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/LSTM.php
                    [2] http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """

    def __init__(self, input, n_in, n_hidden, n_out, reverse=False, corruption_level=0.):

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

        assert self.n_in == self.n_out

        self.input = input

        numpy_rng = np.random.RandomState(123)
        self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.corruption_level = corruption_level
        self.corrupted_input = self.theano_rng.binomial(
            size=self.input.shape, n=1,
            p=1 - corruption_level,
            dtype=theano.config.floatX
        ) * self.input

        def initialize_weights(shape):
            assert len(shape) == 2
            return np.asarray(
                a=np.random.uniform(
                    size=shape,
                    low=-.01,
                    high=.01
                ),
                dtype=theano.config.floatX
            )

        #######################################
        #######################################
        ##### Phase #1: Forward propagate #####
        #######################################
        #######################################

        # parameters for the input gates
        # x: x_{t} - current input frame
        # h: h_{t-1} - output of the previous frame x_{t-1}
        # c: c_{t-1} - cell state (not yet update)
        self.W_xi = theano.shared(value=initialize_weights((n_in, n_hidden)), name='W_xi', borrow=True)
        self.W_hi = theano.shared(value=initialize_weights((n_hidden, n_hidden)), name='W_hi', borrow=True)
        self.W_ci = theano.shared(value=initialize_weights((n_hidden, n_hidden)), name='W_ci', borrow=True)
        self.b_i = theano.shared(value=np.zeros((n_hidden,), dtype=theano.config.floatX), name='b_i', borrow=True)

        # parameters for the forget gates
        self.W_xf = theano.shared(value=initialize_weights((n_in, n_hidden)), name='W_xf', borrow=True)
        self.W_hf = theano.shared(value=initialize_weights((n_hidden, n_hidden)), name='W_hf', borrow=True)
        self.W_cf = theano.shared(value=initialize_weights((n_hidden, n_hidden)), name='W_cf', borrow=True)
        self.b_f = theano.shared(value=np.zeros((n_hidden,), dtype=theano.config.floatX), name='b_f', borrow=True)

        # parameters for the output gates
        # c: c_{t} - cell state (current cell state)
        self.W_xo = theano.shared(value=initialize_weights((n_in, n_hidden)), name='W_xo', borrow=True)
        self.W_ho = theano.shared(value=initialize_weights((n_hidden, n_hidden)), name='W_ho', borrow=True)
        self.W_co = theano.shared(value=initialize_weights((n_hidden, n_hidden)), name='W_co', borrow=True)
        self.b_o = theano.shared(value=np.zeros((n_hidden,), dtype=theano.config.floatX), name='b_o', borrow=True)

        # parameters for computing the values of cells (input)
        self.W_xc = theano.shared(value=initialize_weights((n_in, n_hidden)), name='W_xc', borrow=True)
        self.W_hc = theano.shared(value=initialize_weights((n_hidden, n_hidden)), name='W_hc', borrow=True)
        self.b_c = theano.shared(value=np.zeros((n_hidden,), dtype=theano.config.floatX), name='b_c', borrow=True)

        # initial values for memory cells
        self.c0 = theano.shared(value=np.zeros((n_hidden,), dtype=theano.config.floatX), name='c0', borrow=True)
        self.h0 = T.tanh(self.c0)

        self.params = [
            self.W_xi, self.W_hi, self.W_ci, self.b_i,  # input gates
            self.W_xf, self.W_hf, self.W_cf, self.b_f,  # forget gates
            self.W_xo, self.W_ho, self.W_co, self.b_o,  # output gates
            self.W_xc, self.W_hc, self.b_c,             # input memory cell
            self.c0                                     # initial values of memory cells
        ]

        def lstm_step(x_t, h_tm1, c_tm1):
            # For input, forget and output gates, we usually use the sigmoid function to
            # control "open" or "close".
            # The order of the implementation follows the article "Understanding LSTM Networks -- colah's blog"
            f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf) + T.dot(c_tm1, self.W_cf) + self.b_f)
            i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi) + T.dot(c_tm1, self.W_ci) + self.b_i)
            c_tilde_t = T.tanh(T.dot(x_t, self.W_xc) + T.dot(h_tm1, self.W_hc) + self.b_c)
            c_t = f_t * c_tm1 + i_t * c_tilde_t
            o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo) + T.dot(h_tm1, self.W_ho) + T.dot(c_t, self.W_co) + self.b_o)
            h_t = o_t * T.tanh(c_t)
            return [h_t, c_t]

        [self.h_vals, self.c_vals], _ = theano.scan(
            fn=lstm_step,
            sequences=self.corrupted_input,
            outputs_info=[self.h0, self.c0]
        )

        #################################
        #################################
        ##### Phase #2: Reconstruct #####
        #################################
        #################################

        # parameters for the input gates in reconstruction phase
        self.Wr_hi = theano.shared(value=initialize_weights((n_hidden, n_hidden)), name='Wr_hi', borrow=True)
        self.Wr_ci = theano.shared(value=initialize_weights((n_hidden, n_hidden)), name='Wr_ci', borrow=True)
        self.br_i = theano.shared(value=np.zeros((n_hidden,), dtype=theano.config.floatX), name='br_i', borrow=True)

        # parameters for the forget gates in reconstruction phase
        self.Wr_hf = theano.shared(value=initialize_weights((n_hidden, n_hidden)), name='Wr_hf', borrow=True)
        self.Wr_cf = theano.shared(value=initialize_weights((n_hidden, n_hidden)), name='Wr_cf', borrow=True)
        self.br_f = theano.shared(value=np.zeros((n_hidden,), dtype=theano.config.floatX), name='br_f', borrow=True)

        # parameters for the output gates in reconstruction phase
        self.Wr_ho = theano.shared(value=initialize_weights((n_hidden, n_hidden)), name='Wr_ho', borrow=True)
        self.Wr_co = theano.shared(value=initialize_weights((n_hidden, n_hidden)), name='Wr_co', borrow=True)
        self.br_o = theano.shared(value=np.zeros((n_hidden,), dtype=theano.config.floatX), name='br_o', borrow=True)

        # parameters for computing the values of cells (input) in reconstruction phase
        self.Wr_hc = theano.shared(value=initialize_weights((n_hidden, n_hidden)), name='Wr_hc', borrow=True)
        self.br_c = theano.shared(value=np.zeros((n_hidden,), dtype=theano.config.floatX), name='br_c', borrow=True)

        # parameters for the output (prediction) in reconstruction phase
        self.Wr_out = theano.shared(value=initialize_weights((n_hidden, n_out)), name='Wr_out', borrow=True)
        self.br_out = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='br_out', borrow=True)

        # add the parameters for reconstruction phase
        self.params.extend([
            self.Wr_hi, self.Wr_ci, self.br_i,
            self.Wr_hf, self.Wr_cf, self.br_f,
            self.Wr_ho, self.Wr_co, self.br_o,
            self.Wr_hc, self.br_c,
            self.Wr_out, self.br_out
        ])

        self.Wr_xi = theano.shared(value=initialize_weights((n_out, n_hidden)), name='Wr_xi', borrow=True)
        self.Wr_xf = theano.shared(value=initialize_weights((n_out, n_hidden)), name='Wr_xf', borrow=True)
        self.Wr_xo = theano.shared(value=initialize_weights((n_out, n_hidden)), name='Wr_xo', borrow=True)
        self.Wr_xc = theano.shared(value=initialize_weights((n_out, n_hidden)), name='Wr_xc', borrow=True)
        self.params.extend([self.Wr_xi, self.Wr_xf, self.Wr_xo, self.Wr_xc])

        def lstm_reconstruct_step(x_t, h_tm1, c_tm1):
            f_t = T.nnet.sigmoid(T.dot(x_t, self.Wr_xf) + T.dot(h_tm1, self.Wr_hf) + T.dot(c_tm1, self.Wr_cf) + self.br_f)
            i_t = T.nnet.sigmoid(T.dot(x_t, self.Wr_xi) + T.dot(h_tm1, self.Wr_hi) + T.dot(c_tm1, self.Wr_ci) + self.br_i)
            c_tilde_t = T.tanh(T.dot(x_t, self.Wr_xc) + T.dot(h_tm1, self.Wr_hc) + self.br_c)
            c_t = f_t * c_tm1 + i_t * c_tilde_t
            o_t = T.nnet.sigmoid(T.dot(x_t, self.Wr_xo) + T.dot(h_tm1, self.Wr_ho) + T.dot(c_t, self.Wr_co) + self.br_o)
            h_t = o_t * T.tanh(c_t)
            y_t = T.dot(h_t, self.Wr_out) + self.br_out
            return [y_t, h_t, c_t]

        # Start of reconstructed sequence: act as constant zero vector
        SORS = theano.shared(value=np.zeros((n_in,), dtype=theano.config.floatX), name='SORS', borrow=True)
        [self.yr_vals, self.hr_vals, self.cr_vals], _ = theano.scan(
            fn=lstm_reconstruct_step,
            outputs_info=[SORS, self.h_vals[-1], self.c_vals[-1]],
            n_steps=self.input.shape[0]
        )

        if reverse:
            self.mse = T.mean((self.yr_vals[::-1] - self.input) ** 2)
        else:
            self.mse = T.mean((self.yr_vals - self.input) ** 2)

    def train(self, X_train, X_test, n_epochs, learning_rate, save_steps, feature_range):

        n_train = len(X_train)
        n_test = len(X_test)

        # Linearly scale all values to range [-1, 1]
        f_min, f_max = feature_range
        for i in xrange(n_train):
            X_train[i] = np.asarray(
                a=(X_train[i] - f_min) / (f_max - f_min) * 2 - 1,
                dtype='float32'
            )
        for i in xrange(n_test):
            X_test[i] = np.asarray(
                a=(X_test[i] - f_min) / (f_max - f_min) * 2 - 1,
                dtype='float32'
            )

        print 'compiling theano training functions ...'
        # define the objective function to be minimized
        objective_func = self.mse

        # compute the gradients of the objective function w.r.t. self.params using BPTT
        gparams = T.grad(cost=objective_func, wrt=self.params)

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        # compiling a Theano function `train_model` that returns the
        # cost, but in the same time updates the parameter of the
        # model based on the rules defined in `updates`
        train_model = theano.function(
            inputs=[self.input],
            outputs=objective_func * ((f_max - f_min) ** 2) / 4,
            updates=updates,
            name='train_model'
        )

        test_model = theano.function(
            inputs=[self.input],
            outputs=objective_func * ((f_max - f_min) ** 2) / 4,
            name='test_model'
        )

        if self.corruption_level > 0:
            print '\nBe cautious! The reconstruction error of test_model is computed in terms of noisy input!\n'

        reconstruct_model = theano.function(
            inputs=[self.input],
            outputs=self.yr_vals,
            name='reconstruct_model'
        )

        print 'training the SA model ...'
        # optimize the model parameters using stochastic gradient descent
        train_order = np.arange(n_train)
        for epoch in xrange(n_epochs):
            total_batch_cost = 0.
            print '\nepoch #%d' % (epoch + 1)
            np.random.shuffle(train_order)
            for index in train_order:
                batch_cost = train_model(X_train[index])
                total_batch_cost += batch_cost
            print 'average train batch cost = %f' % (total_batch_cost / n_train)
            test_batch_cost = np.zeros(n_test)
            for index in xrange(n_test):
                test_batch_cost[index] = test_model(X_test[index])
            print 'average test batch cost = %f' % (np.mean(test_batch_cost))
            if (epoch + 1) % save_steps == 0:
                with open('lstmae.' + str(self.n_hidden) + '.' + str(epoch + 1) + '.pkl', 'wb') as f:
                    cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)
        # save the final model
        with open('lstmae.' + str(self.n_hidden) + '.final.pkl', 'wb') as f:
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)
