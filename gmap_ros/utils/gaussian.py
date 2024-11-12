import numpy as np

class Gaussian3:
    def __init__(self, mean, covariance):
        self.mean = mean  
        self.covariance = covariance  

    def eval_log_gaussian(self, sigma_square, delta):
        """Evaluates the log of a Gaussian PDF given variance and delta."""
        if sigma_square <= 0:
            sigma_square = 1e-4   
        return -0.5 * (delta ** 2) / sigma_square - 0.5 * np.log(2 * np.pi * sigma_square)

    def eval(self, p):
        """Evaluates the Gaussian probability for a given OrientedPoint `p`."""
        
        q = p - self.mean 
        # Normalize angle difference
        q[2] = np.arctan2(np.sin(q[2]), np.cos(q[2]))

        # Transform `q` into the eigenvector basis of the covariance
        v1 = self.covariance['evec'][0][0] * q[0] + self.covariance['evec'][1][0] * q[1] + self.covariance['evec'][2][0] * q[2]
        v2 = self.covariance['evec'][0][1] * q[0] + self.covariance['evec'][1][1] * q[1] + self.covariance['evec'][2][1] * q[2]
        v3 = self.covariance['evec'][0][2] * q[0] + self.covariance['evec'][1][2] * q[1] + self.covariance['evec'][2][2] * q[2]

        # Evaluate the log Gaussian for each component and sum them up
        return (self.eval_log_gaussian(self.covariance['eval'][0], v1) +
                self.eval_log_gaussian(self.covariance['eval'][1], v2) +
                self.eval_log_gaussian(self.covariance['eval'][2], v3))

# Example Usage
mean = np.array([0.0, 0.0, 0.0])  
covariance = {
    'eval': [0.5, 0.3, 0.2], 
    'evec': np.identity(3) }

gaussian3 = Gaussian3(mean, covariance)
p = np.array([1.0, 0.5, 0.2])  # Example point to evaluate
print(gaussian3.eval(p))  # Outputs the log probability density for `p`
