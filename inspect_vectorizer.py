import joblib
import traceback

def inspect(path='vectorizer.jb'):
    try:
        obj = joblib.load(path)
        print('Loaded:', path)
        print('Type:', type(obj))
        print('Has attribute transform?', hasattr(obj, 'transform'))
        print('Is transform callable?', callable(getattr(obj, 'transform', None)))
        print('repr (short):', repr(obj)[:800])
    except Exception:
        print('Failed to load or inspect', path)
        print(traceback.format_exc())

if __name__ == '__main__':
    inspect()
# inspect_vectorizer.py
import joblib
import traceback

def inspect(path='vectorizer.jb'):
    try:
        obj = joblib.load(path)
        print('Loaded:', path)
        print('Type:', type(obj))
        print('Has attribute transform?', hasattr(obj, 'transform'))
        print('Is transform callable?', callable(getattr(obj, 'transform', None)))
        print('repr (short):', repr(obj)[:800])
    except Exception:
        print('Failed to load or inspect', path)
        print(traceback.format_exc())

if __name__ == '__main__':
    inspect()
