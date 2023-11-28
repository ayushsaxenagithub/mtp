import csv

# Specify the input and output file paths
input_file_path = 'data.txt'
output_file_path = 'output.csv'

# Read the text file and write to CSV
with open(input_file_path, 'r') as infile, open(output_file_path, 'w', newline='') as outfile:
    # Create a CSV writer object
    csv_writer = csv.writer(outfile)

    # Read each line from the text file
    for line in infile:
        # Split the line into values using multiple spaces as delimiter
        values = line.split()

        # Write the values to the CSV file
        csv_writer.writerow(values)

print(f'Conversion complete. CSV file saved at: {output_file_path}')
