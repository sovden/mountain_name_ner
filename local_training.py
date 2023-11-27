from HF_training import create_datasets, run_training

if __name__ == '__main__':
    datasets, label_list = create_datasets(is_lower_text=True)
    run_training()