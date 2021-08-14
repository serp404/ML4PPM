import os


os.chdir(os.getcwd())


def print_table_and_result(readme_file, params, accuracy, window_size):
    with open(readme_file, 'a') as readme:
        readme.write(f'### Result: {accuracy}\n')
        readme.write('| Parameter | value |\n')
        readme.write('|---|---|\n')
        for key, value in params.items():
            readme.write(f'| {key} | {value} |\n')
        readme.write(f'| window_size | {window_size} |\n')
        readme.write('\n\n\n')
