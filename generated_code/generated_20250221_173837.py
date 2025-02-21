import os

# Create the directory
os.makedirs("1")

# List of file names to create
files = ["file1.txt", "file2.txt", "file3.txt"]

# Create each file in the directory
for file_name in files:
    with open(f"1/{file_name}", "w") as f:
        f.write("")  # Write an empty string to create an empty file
