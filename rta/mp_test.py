from multiprocessing import Process, Queue
from read_in_data import big_data

def worker(tasks):
    while True:
        task = tasks.get()
        if task == "end":
            print("Process got 'end' statement.")
            return
        run, a_run = task
        print(run, id(a_run))


if __name__ == '__main__':
    no_cores = 4
    annotated, unlabelled = big_data(path = "~/Projects/retentiontimealignment/Data/")
    queue = Queue()

    processes = [Process(target=worker, args=(queue,))
                 for _ in range(no_cores)]

    # get some tasks for the processes
    for run, a_run in annotated.groupby('run'):
        queue.put((run, a_run))
        print(run, id(a_run))

    # tell processes that they should get into coma
    for _ in range(no_cores):
        queue.put("end")

    # start processes
    for process in processes:
        process.start()

    # finish processes
    for process in processes:
        process.join()

    print("Finished.")
