import os
import subprocess
import time

from threading import Thread, Lock

class Profiler():

    def __init__(self, log_to_file=False, file_name="profile.txt", debug=False, profile_memory_only=True):
        # Debug parameters
        self.debug = debug

        # File logging parameters
        self.log_to_file = log_to_file
        self.file_name = file_name
        self.logging_file = None

        # If logging to file enabled then open file
        if self.log_to_file:
            self.logging_file = open(self.file_name, "w")

        # Thread parameters
        self.done = False
        self.done_lock = Lock()
        self.profile_memory_only = profile_memory_only

        if self.profile_memory_only:
            self.profile_thread = Thread(group=None, target=self.profile_memory_thread, args=(os.getpid(),))
        else:
            self.profile_thread = Thread(group=None, target=self.profile_thread_function, args=())

    def start_thread(self):
        """
        Function to start the Profiler thread for collecting information.
        """
        self.profile_thread.start()

    def profile_memory(self):
        """
        Function to get the percentage of memory used.
        """
        # Getting all memory using os.popen()
        total_memory, used_memory, free_memory = map(
            int, os.popen('free -t -m').readlines()[-1].split()[1:])

        # Memory usage
        memory_usage = round((used_memory/total_memory) * 100, 2) 

        return memory_usage

    def profile_memory_get_bytes(self):
        """
        Function to get the percentage of memory used.
        """
        # Getting all memory using os.popen()
        total_memory, used_memory, free_memory = map(
            int, os.popen('free -t -m').readlines()[-1].split()[1:])

        return total_memory, used_memory, free_memory

    def profile_energy(self, energy_process):
        """
        Function to get the energy statistics and return the float value.

        @param energy_process: subprocess for calculating the amount of energy.
                               The subprocess must use tegrastats.
        """
        line = energy_process.stdout.readline()
        if line != '':
            line = line.decode('utf-8')
            data = line.split(' ')
            energy = data[data.index('GR3D_FREQ') + 1].strip('%')

            return float(energy)

        return 0.0

    def profile_thread_function(self):
        """
        Thread function to profile the energy and the memory statistics.
        """
        # Create energy subprocess for logging
        energy_process = subprocess.Popen('tegrastats', stdout=subprocess.PIPE)

        if self.debug:
            print("Start profiling.")

        while True:
            self.done_lock.acquire()
            if self.done:
                self.done_lock.release()
                print("Breaking")
                break
            self.done_lock.release()
  
            # Shared Variable
            memory_usage = self.profile_memory()
            energy_usage = self.profile_energy(energy_process)

            usage_str = "RAM %: {}, ENERGY %: {}\n".format(memory_usage, energy_usage)
            if self.debug:
                print(usage_str)

            # Write data to file if option is enabled
            if self.log_to_file:
                self.logging_file.write(usage_str)

            time.sleep(0.1)

        if self.debug:
            print("Stop profiling.")
        
        # Kill energy subprocess
        energy_process.kill()

    def profile_memory_thread(self, pid):
        """
        Function to profile the memory of a process.
        """

        PS_MEMORY_INDEX = 3

        while True:
            self.done_lock.acquire()
            if self.done:
                self.done_lock.release()
                break
            self.done_lock.release()

            info_process = subprocess.Popen(['ps', '-u', '-p', str(pid)], stdout=subprocess.PIPE)

            # Line 1 is of no use so skip
            info_process.stdout.readline()

            # Line 2 is of use because it has the statistics
            line = info_process.stdout.readline().decode('utf-8')

            # Get memory usage in percentage
            memory_usage = line.split()[PS_MEMORY_INDEX]

            # Log to file
            if self.log_to_file:
                self.logging_file.write("{}\n".format(memory_usage))

            time.sleep(0.1)

    def stop_thread(self):
        """
        Function to stop the profiler. Waits for the
        profiler thread to finish.
        """
        self.done_lock.acquire()
        self.done = True
        self.done_lock.release()

        if self.debug:
            print("Stopping profiler.")
            print("Waiting for the thread to join.")

        # Wait for the thread to finish
        self.profile_thread.join()

        # Close the logging file if required
        if self.log_to_file:
            self.logging_file.close()

if __name__ == "__main__":
    """
    Testing.
    """
    profiler = Profiler(log_to_file=False, debug=True, profile_memory_only=False)
    profiler.start_thread()
    time.sleep(10)
    profiler.stop_thread()
