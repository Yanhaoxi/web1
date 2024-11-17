import csv
from process_data import read_csv, token2tags, jiebaModel

csv_file_path = 'selected_movie_top_1200_data_tag.csv'
output_csv_file_path = 'processed_selected_movie_top_1200_data_tag.csv'


if __name__ == '__main__':
    data = read_csv(csv_file_path, 1200)
    result = []
    model = jiebaModel()
    i=0
    for movie_id, tags in data:
        result.append((movie_id, token2tags(tags, model)))
        i+=1
        if i % 100 == 0:
            print(f'Processed {i} movies.')
    # 写回新的 CSV 文件
    with open(output_csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['book', 'tags'])  # 写入表头
        writer.writerows(result)