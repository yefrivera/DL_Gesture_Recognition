def load_gestures(path='dataset/gestures.txt'):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]
