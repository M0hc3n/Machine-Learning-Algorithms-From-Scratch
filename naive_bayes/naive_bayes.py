import numpy as np

from utils.Pdf import PDF

# this class implements the Naive Bayes algorithm
# check out this rich article for more information
# https://towardsdatascience.com/all-about-naive-bayes-8e13cef044cf
class NaiveBayes:

    # an implementation of the fit method
    def fit(self, X,y):
        n_sample, n_features = X.shape

        # extract all the target classes
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # initialize the mean, var, and the priors (P(c) with c being a class) of each class 
        self.means = np.zeros((n_classes, n_features), dtype=np.float64)
        self.variances = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)


        # calculete the mean, var and P(c) for each class c
        for idx, c in enumerate(self.classes):
            # extract all samples from the dataset having c as a class
            X_c = X[ y == c ]

            # (axis = 0) indicates here to calculate the mean for each column
            self.means[idx,:] = X_c.mean(axis=0)
            self.variances[idx,:] = X_c.var(axis=0)

            # P(c) = number of samples having class c / total size of the dataset
            self.priors[idx] = X_c.shape[0] / float(n_sample) # recall that shape's first element in
                                                         # the tuple is the number of sample (i.e. size)

    def predict(self, X_sample):
        y_hat = [self.predict_helper(x) for x in X_sample]

        return np.array(y_hat)

    def predict_helper(self,x):
        # initialize the array of P(c/X), so called posteriors
        posteriors = []

        # recall that for each class we need to calculate the following
        # posterior(c) = sum_of( log( P(X_i / c) ) ) + log ( P(y) )
        #                         > (above) also called priors
        for idx, c in enumerate(self.classes):
            # recall that we use log to avoid float overflow when doing the product operation
            prior = np.log(self.priors[idx])

            mean_of_class_c = self.means[idx]
            variance_of_class_c = self.variances[idx]

            # returns a vector P(X / y) formed of ( P(X_0 / y), P(X_1 / y) ... P(X_n / y)  )
            # where n is the number of features
            pdf_value = PDF().calculate_gaussian_density(x, mean_of_class_c, variance_of_class_c)

            posterior = np.sum(np.log(pdf_value))
            posterior += prior

            # add the calculated value to our existing list of posteriors
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posterior)]

            
     




