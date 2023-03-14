import os
import csv
import numpy as np


class Logger:
    def __init__(self, dataset, architecture, test_iteration=0):
        # init
        if(not os.path.exists(__file__.replace('\\', '/')[:__file__.rfind('\\')+1]+'logs/')):
            os.mkdir(__file__.replace('\\', '/')
                     [:__file__.rfind('\\')+1]+'logs/')
        if(not os.path.exists(__file__.replace('\\', '/')[:__file__.rfind('\\')+1]+'logs/'+dataset+'/')):
            os.mkdir(__file__.replace('\\', '/')
                     [:__file__.rfind('\\')+1]+'logs/'+dataset)
        if(not os.path.exists(__file__.replace('\\', '/')[:__file__.rfind('\\')+1]+'logs/'+dataset+'/'+architecture+'/')):
            os.mkdir(__file__.replace('\\', '/')
                     [:__file__.rfind('\\')+1]+'logs/'+dataset+'/'+architecture+'/')

        self.dataset = dataset
        self.architecture = architecture
        self.path = __file__.replace(
            '\\', '/')[:__file__.rfind('\\')+1]+'logs/'+dataset+'/'+architecture+'/'

        # create new dir
        while(os.path.exists(self.path+str(test_iteration))):
            test_iteration += 1
        self.path += str(test_iteration)+'/'
        os.mkdir(self.path)
        open(self.path+'test.csv', "x")
        open(self.path+'train.csv', "x")
        open(self.path+'time.csv', "x")

        # create file objects
        test_file_write = open(self.path+'test.csv', "w", newline="")
        train_file_write = open(self.path+'train.csv', "w", newline="")
        time_file_write = open(self.path+'time.csv', "w", newline="")

        # create csvwriter object
        test_writer = csv.writer(test_file_write)
        train_writer = csv.writer(train_file_write)
        time_writer = csv.writer(time_file_write)

        # write headers
        header = ['iter', 'acc', 'samples']
        test_writer.writerow(header)
        train_writer.writerow(header)
        header = ['iter', 'time', 'samples']
        time_writer.writerow(header)

        # write initial data
        data = [[i, 0, 0] for i in range(3000)]
        test_writer.writerows(data)
        train_writer.writerows(data)
        time_writer.writerows(data)

        # close files
        test_file_write.close()
        train_file_write.close()
        return

    def __del__(self):
        return

    def train_acc_at_iteration(self, iter, accuracy):
        f_read = open(self.path+'train.csv')
        reader = csv.reader(f_read)
        header = []
        rows = []

        # read
        header = next(reader)
        for row in reader:
            rows.append([int(row[0]), float(row[1]), int(row[2])])
        f_read.close()

        # calculate new average
        rows[iter][1] = (rows[iter][1]*rows[iter]
                         [2]+accuracy)/(rows[iter][2]+1)
        rows[iter][2] += 1

        # write
        f_write = open(self.path+'train.csv', 'w', newline="")
        writer = csv.writer(f_write)
        writer.writerow(header)
        writer.writerows(rows)
        f_write.close()
        return

    def test_acc_at_iteration(self, iter, accuracy):
        f_read = open(self.path+'test.csv')
        reader = csv.reader(f_read)
        header = []
        rows = []

        # read
        header = next(reader)
        for row in reader:
            rows.append([int(row[0]), float(row[1]), int(row[2])])
        f_read.close()

        # calculate new average
        rows[iter][1] = (rows[iter][1]*rows[iter]
                         [2]+accuracy)/(rows[iter][2]+1)
        rows[iter][2] += 1

        # write
        f_write = open(self.path+'test.csv', 'w', newline="")
        writer = csv.writer(f_write)
        writer.writerow(header)
        writer.writerows(rows)
        f_write.close()
        return

    def train_time_at_iteration(self, iter, time):
        f_read = open(self.path+'time.csv')
        reader = csv.reader(f_read)
        header = []
        rows = []

        # read
        header = next(reader)
        for row in reader:
            rows.append([int(row[0]), float(row[1]), int(row[2])])
        f_read.close()

        # calculate new average
        rows[iter][1] = (rows[iter][1]*rows[iter]
                         [2]+time)/(rows[iter][2]+1)
        rows[iter][2] += 1

        # write
        f_write = open(self.path+'time.csv', 'w', newline="")
        writer = csv.writer(f_write)
        writer.writerow(header)
        writer.writerows(rows)
        f_write.close()
        return
