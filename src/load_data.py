from datasets import load_dataset

def get_imdb():
    return load_dataset("stanfordnlp/imdb")

if __name__ == "__main__":
    ds = get_imdb()
    print(ds)
