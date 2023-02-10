import os
import glob
from tqdm.auto import tqdm

def concatenate_files(books: list, outfile_name: str):
    """
    Concatenates all books txts into a single large txt file
    """
    with open(outfile_name, 'w') as outfile:
        for book in tqdm(books):
            with open(book) as infile:
                for line in infile:
                    outfile.write(line)
    print("Done!")

if __name__ == "__main__":
    books = glob.glob("data/*.txt")
    outfile_name = "data/all_books.txt"
    concatenate_files(books, outfile_name)