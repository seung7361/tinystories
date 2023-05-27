# TinyStories
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)
![OpenAI GPT-3](https://img.shields.io/badge/OpenAI-GPT--3-red.svg)

TinyStories is a Python project that uses OpenAI's GPT-3 model to generate creative short stories. Given an initial text or a prompt, the model creates an engaging short story around it.

## Usage

The model is extremely straightforward to use! Just call the `generate` function of the model with your text.

```python
from tinystories import TinyStoriesModel

model = TinyStoriesModel()
story = model.generate("<Your Text Here>")
print(story)
```
Replace "<Your Text Here>" with the text or prompt you want to generate a story from.

## Requirements
To run this project, you need the following:

Python 3.8 or higher
Transformers library

## Installation

1. Clone the repository:
```
git clone https://github.com/<YourUserName>/tinystories.git
```

2. Navigate into the cloned repository:
```
cd tinystories
```

3. Install the requirements:
```
pip install -r requirements.txt
```

## Licensing
TinyStories is open source software ![licensed as MIT]().