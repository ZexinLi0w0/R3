import sys

def get_max_memory(file_name):
    """
    Function to find the max memory usage.

    @param file_name: Name of the file to read memory usage data from.
    """
    memory_usage_file = open(file_name, "r")
    lines = memory_usage_file.readlines()

    data = [float(line) for line in lines]
    print(max(data))

    memory_usage_file.close()

if __name__ == "__main__":
    file_name = "profile.txt"
    if len(sys.argv) == 2:
        file_name = sys.argv[1]
    get_max_memory(file_name)
