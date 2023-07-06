import numpy as np

class Score:

    # utility method to calculate the r_squared score
    # here is a good article about the implementation: 
    # https://www.askpython.com/python/coefficient-of-determination
    def r2_score(self, y_true, y_hat):
        # we need to start by caculating the correlation matrix
        # to grab the correlation value between y_true and y_hat
        corr_mat = np.corrcoef(y_true, y_hat)

        # now, we know that r2_score = (correlation between y_true and y_hat)^2
        # since the correlation matrix has the following form
        # [
        #   [
        #       correlation between y_true and y_true, 
        #       correlation between y_true and y_hat
        #   ],
        #   [
        #       correlation between y_hat and y_true,
        #       correlation between y_true and y_true 
        #   ]
        # ]

        correlation = corr_mat[0,1] # yes, we could also do: corr_mat[1,0]

        return correlation**2 

