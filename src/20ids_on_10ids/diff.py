import difflib
import os

# Specify the directory where your text files are located
directory = './'

# Get a list of text files in the directory
text_files = [f for f in os.listdir(directory) if f.endswith('.txt') and f.startswith('logs')]

# Compare each pair of text files
for i in range(len(text_files)):
    for j in range(i + 1, len(text_files)):
        file1 = os.path.join(directory, text_files[i])
        file2 = os.path.join(directory, text_files[j])

        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()

        # Calculate the difference between the two files
        d = difflib.Differ()
        diff = list(d.compare(lines1, lines2))

        # Print the differences
        print(f"Differences between {text_files[i]} and {text_files[j]}:")
        for line in diff:
            if line.startswith('- ') or line.startswith('+ '):
                print(line)

        print("\n")
