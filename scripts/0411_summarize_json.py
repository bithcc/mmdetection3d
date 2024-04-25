import json
import pandas as pd
from tqdm import tqdm

# Load the JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

# Process the JSON data to extract performance metrics
def process_data(data):
    # Initialize a dictionary to hold all data
    all_data = {}
    for entry in tqdm(data):
        for key, value in entry.items():
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(value)
    return all_data

# Save the processed data to an Excel file
def save_to_excel(data, output_path):
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    # Convert dictionary to DataFrame and write to Excel
    df = pd.DataFrame(data)
    df.to_excel(writer, index=False, sheet_name='Performance Metrics')
    writer.save()

def main():
    file_path = '/path/to/your/20240322_145420.json'  # Update the path to your JSON file
    output_path = '/path/to/output/performance_metrics.xlsx'  # Desired output path for the Excel file

    json_data = load_json_data(file_path)
    processed_data = process_data(json_data)
    save_to_excel(processed_data, output_path)

if __name__ == '__main__':
    main()
