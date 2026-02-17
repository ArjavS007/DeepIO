from _1_csv_files_extraction import ExtractCsvFiles
from _2_bft_data_json import BftDataJson
from _3_csv_filtering import CSVFiltering
from _4_plots import Plots
from _5_process_csvs import ProcessCSV


def extract_and_visualize(raw_data_dir, csv_data_dir):
    """Run extraction + filtering + visualization pipeline"""
    print(">>> Running data extraction and visualization...")
    ExtractCsvFiles(base_dir=raw_data_dir, target_dir=csv_data_dir).main()
    BftDataJson(csv_file_path=csv_data_dir).main()
    CSVFiltering(csv_file_path=csv_data_dir).main()
    Plots(csv_file_path=csv_data_dir).main()
    ProcessCSV(base_dir=csv_data_dir).main()
    print(">>> Extraction and visualization done.")


if __name__ == "__main__":
    raw_data_dir = "/home/ubuntu/DeepIO/switch/data/MPGCS"
    csv_data_dir = "/home/ubuntu/DeepIO/switch/data/csvs"
    extract_and_visualize(raw_data_dir, csv_data_dir)
