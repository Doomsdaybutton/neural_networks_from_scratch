import os
import pandas as pd
import numpy as np


class Logger:
    def __init__(self, learning_rate, mini_batch_size, epochs, cost_function):
        counter = 0
        self.path_dir = __file__[:__file__.rfind(os.sep)+1]+"logs"+os.sep

        while(os.path.exists(self.path_dir+str(counter))):
            counter += 1

        self.counter = counter
        self.path_dir = self.path_dir+str(counter)+os.sep
        os.mkdir(self.path_dir)
        self.accuracy_at_iteration = []
        self.time_per_iteration = 0
        self.time_per_iteration_samples = 0
        open(self.path_dir+"accuracy_at_iteration.csv", "x")

        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.cost_function = cost_function

        # create files that hold the mean values over all testings
        if not (os.path.isfile(os.path.normpath(self.path_dir+os.path.pardir+os.sep+"accuracy_at_iteration_mean.csv"))):
            open(os.path.normpath(self.path_dir+os.path.pardir +
                 os.sep+"accuracy_at_iteration_mean.csv"), "w").write("iteration,mean_accuracy_with_mean_squared_error,mean_accuracy_with_cross_entropy,mean_squared_error_samples,cross_entropy_samples")
        if not (os.path.isfile(os.path.normpath(self.path_dir+os.path.pardir+os.sep+"time_per_iteration_mean.csv"))):
            open(os.path.normpath(self.path_dir+os.path.pardir +
                 os.sep+"time_per_iteration_mean.csv"), "w").write("mean_time_per_iteration_with_mean_squared_error,mean_time_per_iteration_with_cross_entropy,mean_squared_error_samples,cross_entropy_samples")

    def log_accuracy_at_iteration(self, iteration, accuracy):
        self.accuracy_at_iteration.append(
            [iteration, accuracy])

    def log_time_per_iteration(self, time):
        self.time_per_iteration_samples += 1
        self.time_per_iteration += time

    def close(self):
        self.time_per_iteration = self.time_per_iteration/self.time_per_iteration_samples
        accuracy_at_iteration = pd.DataFrame(
            self.accuracy_at_iteration, columns=["iteration", "accuracy"])
        accuracy_at_iteration.to_csv(
            self.path_dir+"accuracy_at_iteration.csv", index=False)
        accuracy_at_iteration_mean = pd.read_csv(os.path.normpath(
            self.path_dir+os.pardir+os.sep+"accuracy_at_iteration_mean.csv"))
        time_per_iteration_mean = pd.read_csv(os.path.normpath(
            self.path_dir+os.pardir+os.sep+"time_per_iteration_mean.csv"))
        accuracy_at_iteration_mean = np.array(
            accuracy_at_iteration_mean, dtype=float)
        time_per_iteration_mean = np.array(
            time_per_iteration_mean, dtype=float)

        # correct mean values
        if(self.cost_function == "mean_squared_error"):
            time_per_iteration_mean[0, 2] += 1
            time_per_iteration_mean[0, 0] = (time_per_iteration_mean[0, 0]*(
                time_per_iteration_mean[0, 2]-1)+self.time_per_iteration)/time_per_iteration_mean[0, 2]

            for i in range(len(self.accuracy_at_iteration)):
                accuracy_at_iteration_mean[i, 0] = i
                accuracy_at_iteration_mean[i, 3] += 1
                accuracy_at_iteration_mean[i, 1] = (accuracy_at_iteration_mean[i, 1]*(
                    accuracy_at_iteration_mean[i, 3]-1)+self.accuracy_at_iteration[i][1])/accuracy_at_iteration_mean[i, 3]

        if(self.cost_function == "cross_entropy"):
            time_per_iteration_mean[0, 3] += 1
            time_per_iteration_mean[0, 1] = (time_per_iteration_mean[0, 1]*(
                time_per_iteration_mean[0, 3]-1)+self.time_per_iteration)/time_per_iteration_mean[0, 3]

            for i in range(len(self.accuracy_at_iteration)):
                accuracy_at_iteration_mean[i, 0] = i
                accuracy_at_iteration_mean[i, 4] += 1
                accuracy_at_iteration_mean[i, 2] = (accuracy_at_iteration_mean[i, 2]*(
                    accuracy_at_iteration_mean[i, 4]-1)+self.accuracy_at_iteration[i][1])/accuracy_at_iteration_mean[i, 4]

        time_per_iteration_mean_csv = pd.DataFrame(time_per_iteration_mean, columns=[
                                                   "mean_time_per_iteration_with_mean_squared_error", "mean_time_per_iteration_with_cross_entropy", "mean_squared_error_samples", "cross_entropy_samples"])
        accuracy_at_iteration_mean_csv = pd.DataFrame(
            accuracy_at_iteration_mean, columns=["iteration", "mean_accuracy_with_mean_squared_error", "mean_accuracy_with_cross_entropy", "mean_squared_error_samples", "cross_entropy_samples"])
        time_per_iteration_mean_csv.to_csv(os.path.normpath(
            self.path_dir+os.path.pardir+os.sep+"time_per_iteration_mean.csv"), index=False)
        accuracy_at_iteration_mean_csv.to_csv(os.path.normpath(
            self.path_dir+os.path.pardir+os.sep+"accuracy_at_iteration_mean.csv"), index=False)


if __name__ == "__main__":
    logger = Logger()
    logger.log_accuracy_at_iteration(1, 3)
    logger.log_accuracy_at_iteration(2, 6)
    logger.log_accuracy_at_iteration(3, 8)
    print(logger.accuracy_at_iteration)
    logger.close()
