import os


class Logger:
    def __init__(self, dataset, architecture, test_iteration=0):
        self.dataset = dataset
        self.architecture = architecture
        self.path = __file__.replace(
            '\\', '/')[:__file__.rfind('\\')+1]+'logs/'+dataset+'/'+architecture+'/'
        while(os.path.exists(self.path+str(test_iteration))):
            test_iteration += 1
        self.path += str(test_iteration)
        os.mkdir(self.path)
        return

    def train_acc_at_iteration(self, iteration, accuracy):
        print("Train: Iteration:{} Accuracy:{}".format(iteration, accuracy))
        return

    def test_acc_at_iteration(self, iteration, accuracy):
        print("Test: Iteration:{} Accuracy:{}".format(iteration, accuracy))
        return


log = Logger('mnist', 'architecture1')
