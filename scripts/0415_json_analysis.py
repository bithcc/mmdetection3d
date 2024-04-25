import json
import pandas as pd
from tqdm import tqdm
from openpyxl import Workbook
from openpyxl.styles import Font

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
        if 'miou' in entry:
            for key, value in entry.items():
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(value)
    return all_data

# Save the processed data to an Excel file
def save_to_excel(data, output_path):
    # Convert dictionary to DataFrame
    df = pd.DataFrame(data)
    
    # Create a Pandas Excel writer using openpyxl as the engine
    writer = pd.ExcelWriter(output_path, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Performance Metrics')
    
    # Load the openpyxl workbook object
    workbook = writer.book
    worksheet = writer.sheets['Performance Metrics']
    
    # Apply bold font to the maximum value in each column
    for col in worksheet.columns:
        # Ignoring the header row (column names)
        column = [cell.value for cell in col[1:]]
        max_value = max(column)
        for cell in col[1:]:  # Skip header row for formatting
            if cell.value == max_value:
                cell.font = Font(bold=True)

    # Save the workbook
    writer.close()

def main():
    file_path = '/home/ps/huichenchen/mmdetection3d/results2/cylinder3d/0422-parallel-noddcm/20240423_025709/vis_data/20240423_025709.json'  # Update the path to your JSON file
    output_path = '/home/ps/huichenchen/mmdetection3d/results2/analysis/0422_parallel-noddcm.xlsx'  # Desired output path for the Excel file

    json_data = load_json_data(file_path)
    processed_data = process_data(json_data)
    save_to_excel(processed_data, output_path)

if __name__ == '__main__':
    main()




    
