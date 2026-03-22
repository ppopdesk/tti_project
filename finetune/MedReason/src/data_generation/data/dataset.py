from datasets import load_dataset
from torch.utils.data import Dataset

# now get a parser class for the dataset, unify the getting of data interface
class QADataset(Dataset):
    def __init__(self, file_type, path, parsers, split = 'train', **kwargs):
        if file_type == 'huggingface':
            self.ds = load_dataset(path, **kwargs)[split]
        else:
            self.ds = load_dataset(file_type, data_files=path)[split]
        self.parsers = parsers
        
    def __len__(self):
        return len(self.ds)
    
    def default_parser(self, keys: list[dict], row: dict):
        return ' '.join(part['prefix'] + row[part['key']] + part['suffix'] for part in keys)
    
    
    # "choices": ["six weeks post-fertilization.", "eight weeks post-fertilization.", "ten weeks post-fertilization.", "twelve weeks post-fertilization."]
    def mmlu_option_parser(self, row: dict):
        choices = row['choices']
        context = '\n'.join([f'{chr(ord("A") + i)}. {choice}' for i, choice in enumerate(choices)])
        return "Answer Choices:\n" + context
    
    # "options": {"A": "Epistaxis", "B": "Permanent loss of smell", "C": "Persistent nasal crusting", "D": "Persistent congestion"}
    def medqa_option_parser(self, row: dict):
        options = row['options']
        context = '\n'.join([f'{key}. {option}' for key, option in options.items()])
        return "Answer Choices:\n" + context
    
    # there are 4 columns for options, opa, opb, opc, opd
    def medbullets_op4_option_parser(self, row: dict):
        options = [row['op' + chr(ord('a') + i)] for i in range(4)]
        context = '\n'.join([f'{chr(ord("A") + i)}. {option}' for i, option in enumerate(options)])
        return "Answer Choices:\n" + context
    
    # there are 5 columns for options, opa, opb, opc, opd, ope
    def medbullets_op5_option_parser(self, row: dict):
        options = [row['op' + chr(ord('a') + i)] for i in range(5)]
        context = '\n'.join([f'{chr(ord("A") + i)}. {option}' for i, option in enumerate(options)])
        return "Answer Choices:\n" + context
    
    # there are 4 columns for options, opa, opb, opc, opd
    def medmcqa_option_parser(self, row: dict):
        options = [row['op' + chr(ord('a') + i)] for i in range(4)]
        context = '\n'.join([f'{chr(ord("A") + i)}. {option}' for i, option in enumerate(options)])
        return "Answer Choices:\n" + context
    
    # "options": {"A": "Epistaxis", "B": "Permanent loss of smell", "C": "Persistent nasal crusting", "D": "Persistent congestion"}
    def medxpertqa_option_parser(self, row: dict):
        options = row['options']
        context = '\n'.join([f'{key}. {option}' for key, option in options.items()])
        return "Answer Choices:\n" + context
    
    # two options, yes or no
    def pubmedqa_option_parser(self, row: dict):
        return "Answer Choices:\nA. Yes\nB. No"
    
    def medxpertqa_answer_parser(self, row: dict):
        options = row['options']
        answer_label = row['label']
        
        return f'({answer_label}) ' + options[answer_label]
    
    def mmlu_answer_parser(self, row: dict):
        answer = row['answer']
        choice = row['choices']
        
        assert len(choice) == 4
        answers_id = [ord(ans) - ord('a') for ans in answer]
        
        result = ' And '.join([choice[id] for id in answers_id])
        return result
    
    def medmcqa_answer_parser(self, row: dict):
        answer_id = row['cop']
        # there is four choices columns, opa, opb, opc, opd
        result = row['op' + chr(ord('a') + answer_id)]
        
        exp = row['exp']
        if exp is not None:
            result = result + '. Explanation: ' + exp
        return result
        
    def __getitem__(self, idx):
        # read the raw row
        raw_data = {}
        for key in self.ds.column_names:
            raw_data[key] = self.ds[key][idx]
            
        result = {}
        
        for component in ['question', 'answer', 'comparison','options']:
            parser = self.parsers[component]
            # if parser is a list, use the default parser
            if isinstance(parser, list):
                result[component] = self.default_parser(parser, raw_data)
            # else the parser is a string, call the function with the string name, which is a function in the class
            elif isinstance(parser, str):
                func = getattr(self, parser)
                result[component] = func(raw_data)
        return result
