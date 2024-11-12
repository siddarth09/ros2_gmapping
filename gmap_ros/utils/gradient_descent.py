import numpy as np

class OptimizerParams:
    def __init__(self, discretization, angular_step, linear_step, iterations, max_range):
        self.discretization = discretization
        self.angular_step = angular_step
        self.linear_step = linear_step
        self.iterations = iterations
        self.max_range = max_range

class Optimizer:
    def __init__(self, params, likelihood, map_obj):
        self.params = params
        self.likelihood = likelihood
        self.lmap = map_obj

    def gradient_descent(self, old_reading, new_reading):
        self.lmap.clear()
        self.lmap.update(old_reading, OrientedPoint(0, 0, 0), self.params.max_range)
        
        delta = absolute_difference(new_reading.get_pose(), old_reading.get_pose())
        best_pose = delta
        best_score = self.likelihood(self.lmap, new_reading, best_pose, self.params.max_range)
        
        it = 0
        lstep = self.params.linear_step
        astep = self.params.angular_step
        
        while it < self.params.iterations:
            increase = False
            it_best_pose = best_pose
            it_best_score = best_score
            
            while True:
                it_increase = False
                test_best_pose = it_best_pose
                test_best_score = it_best_score
                
                for move in ['Forward', 'Backward', 'Left', 'Right', 'TurnRight', 'TurnLeft']:
                    test_pose = OrientedPoint(it_best_pose.x, it_best_pose.y, it_best_pose.theta)
                    
                    if move == 'Forward':
                        test_pose.x += lstep
                    elif move == 'Backward':
                        test_pose.x -= lstep
                    elif move == 'Left':
                        test_pose.y += lstep
                    elif move == 'Right':
                        test_pose.y -= lstep
                    elif move == 'TurnRight':
                        test_pose.theta -= astep
                    elif move == 'TurnLeft':
                        test_pose.theta += astep
                    
                    score = self.likelihood(self.lmap, new_reading, test_pose, self.params.max_range)
                    if score > test_best_score:
                        test_best_score = score
                        test_best_pose = test_pose
                
                if test_best_score > it_best_score:
                    it_best_score = test_best_score
                    it_best_pose = test_best_pose
                    it_increase = True
                else:
                    break

            if it_best_score > best_score:
                best_score = it_best_score
                best_pose = it_best_pose
                increase = True
            else:
                it += 1
                lstep *= 0.5
                astep *= 0.5
        
        return best_pose
    