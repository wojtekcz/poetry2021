import random

# Sample chunk_len token-sized chunks to a file

# chunk_len = 100 #400
# def random_chunk():
#     start_index = random.randint(0, file_tok_len - chunk_len -1)
#     end_index = start_index + chunk_len + 1
#     return file_tok[start_index:end_index]

flatten = lambda t: [item for sublist in t for item in sublist]


class LineChunker:
    def __init__(self, file_tok: [str], chunk_len: int):
        file_str = " ".join(file_tok)
        file_lines = [(x.strip() + ' _eol_').strip() for x in file_str.split('_eol_')]
        self.file_lines_tok = [x.split() for x in file_lines]
        self.chunk_len = chunk_len
        self.last_num_lines = self.count_tok_lines(self.file_lines_tok[::-1], chunk_len=self.chunk_len)
        self.last_line_index = len(self.file_lines_tok) - self.last_num_lines

    @staticmethod
    def count_tok_lines(file_lines_tok: [[str]], chunk_len: int):
        # count how many last lines (almost) add up to chunk_len
        sum_tok = 0
        idx = 0
        while True:
            n = len(file_lines_tok[idx])
            if sum_tok+n >= chunk_len or idx+1 >= len(file_lines_tok):
                break
            sum_tok += n
            idx += 1

        return idx

    def random_chunk(self):
        start_index = random.randint(0, self.last_line_index)
        num_lines = self.count_tok_lines(self.file_lines_tok[start_index:], chunk_len=self.chunk_len)
        end_index = start_index + num_lines
        return flatten(self.file_lines_tok[start_index:end_index])
