# mountain_name_ner
# About
Repository for solving NER task for mountain entity ("Last year I climbed on Hoverla and it was fun" -> Hoverla - mountain). Training was done with Hugging Face transformers lib. 
# Dataset:
conll2003 ner dataset was used as reference (and for saturation/diversity of custom dataset) https://www.kaggle.com/datasets/juliangarratt/conll2003-dataset. Main idea of creation of dataset is next:
1. Create list of all mountains on earth.
2. Parse internet and generate sentences (using created list).
3. Preprocess collected data (clean, lower, etc). Split into sentences.
4. Take only sentences with mountain name from created list (taking that one without is dangerous, because it could be mistake of automatic searching and sentece actualy does have mountain name and if we take this sentence we just label it as without entity one)
5. Split sentences into words, label each word with ner-tag. Handle 2+ words names. **'O'** - non mountain, **'B-MON'** - first part of 2+ words mountain name, **'I-MON'** - 1 word mountain name or non first parts of 2+ words mountain names.
6. Format sentences like conll2003 dataset ```clean_data/dataset_mountains.txt```. Create csv table for easy analysis ```clean_data/dataset_mountains.csv```.
7. Before training, add examples without entity name from the conll2003 dataset. This dataset almost doesn't contain mountain names. For simplicity, just part of this data were used and relabeled with **'O'**. Proportion original_data/conll2003 1:2 or 1:3 (could be modified during training).
8. Previous point could be considered as augumentaion step. In this case code should be changed, becuase test set is generated from the custom data + conll2003 as well as train set. However, it was decided that this is not augumentaion step, because conll2003 is not something generated from original data, but original itself. 

## Raw data (collected data):
This folder ```data_preparation/raw_data``` contains all data that was collected for this task.For this task there were collected 4 types of data:
1. wikipedia: ```list_of_climbings_wikipedia.txt```, ```list_of_earth_mountains.csv```
2. hiking forums: ```hiking_forums.txt```
3. science texts: ```science_texts.txt```
4. chat gpt generations: ```chatgpt_1.txt```, ```chatgpt_2.txt```

Wikipedia and hiking forums were parsed by with notebook: https://github.com/sovden/mountain_name_ner/blob/master/data_preparation/parsing.ipynb.
Science text was copied manualy from pdf book as well as chat gpt generation.

### Possible improvements:
1. Parse more wikipedia data (generally random article contains only a few sentences with moutain name even for related topics, so amount of article should very big for for noticeable improvement).
2. Parse more hiking forums. This type of source provide quite clean "saturated" with mountain names texts. But 2 problems: create list of forums/pages, create different parsing/cleaning for each.
3. Parse more science texts. Should be quite easy, only list of related articles/books/etc is required.
4. Intensive rephrasing with chatgpt or some LLM.
5. Using of science fiction. Dangerous, because of possible fiction mountain names.

## Data Preparation:
data_preparation/assamble_raw_data.ipynb makes all work. Main steps:
1. One example = one sentence
4. Mountain list cleaning. Shorting of mountain names:
> It turned out that a lot of mountains have name of format MOUNT %FAMOUS LOCATION% (i.e. MOUNT WASHINGTON), while some other names are equally often used with and without word MOUNT (EVEREST/MOUNT EVETEST). In this task, it was decided manualy define list of names with mandotary word MOUNT to avoid confusing, while for every other delete this prefix.
5. Text cleaning steps:
- str.strip();
- ```symbols_to_replace = [["Ã¼","u"], ["Ã¶", "o"],["â€“","-"],["Ã©","e"], ["Ã³","o"], ["Â°","*"]]```;
- clean everything in brackets (Wikipedia and science texts have a lot of them and mostly it just some footnotes or some unuseful details;
- lower all text for simplicity;
- stemming and lemmatization were avoided because conll2003 dataset looks quite raw in these terms;
6. Filter all sentences with less than 25 chars minus length of mountain name (based on histogram for less than 400 chars sentences):
- ![image](https://github.com/sovden/mountain_name_ner/assets/152089125/0efef3f7-331f-4952-8075-64b606ed65e4)
7. Tag creation. For each word in sentence: **'O'** - non mountain, **'B-MON'** - first part of 2+ words mountain name, **'I-MON'** - 1 word mountain name or non first parts of 2+ words mountain names.
8. Check data for consistency (number of tags = number of words; 2+ words names = 2+ tags 
9. Save data in conll2003 dataset format.

### Possible improvements:
1. Improve logic of mountain name shorting, try to avoid list with exclusions.
2. Add **'E-MON'** end tag for long name mountains
3. Create clean Test dataset
4. Add some kind of Augmentation

# Training:
### Code:
```run_training``` from the ```HF_training.py``` provides possibility to run training of one model. Playing with arguments (adding new as well as changing) provide possibility to perform hyperparameter tuning. Currently implemented: pretrained ```model_name``` from HF hub, ```num_of_epochs```, ```lerning_rate```, type of HF ```sheduler```
### Results:
After a few training with different parameters ```albert-base-v2``` with uncased dataset looks like one of the best. Probably small uncased model performs the best becaus of small dataset. Preptrained checkpoint could be found here ```chekpoints_to_upload/albert-base-v2/best_checkpoint```
### Possible Improvements:
1. Try some ner pretrained models.
2. If enough time and resources - run hyperparameter tunnning.
3. Rewrite training using Pytorch/PytochLightning (initialize HF model with nn.Module).
# Inference:
For Inference and playing with model you need to install transformers lib. You can use ```inference_util.py``` or notebook ```inference.ipynb```.
### Examples:
- input: ```"flying over Africa we saw how the top of Naki looks above the clouds"```
- output: ```Named Entities ['nak', 'i']```
### Possible improvements:
1. Implement function that will identify word, not tokens: Named Entities ```['Naki']```.
2. Implement simple API (flask?) for playing with model.
# Local run
## training:
You can just use ```local_training.py``` playing with arguments of dataset and training
## inference:
Inference is based on the ```inference_util.py``` 
> **!!!** if You use cased model for inference, initialize pretrained model with ```start_model = PrepareModel(is_lower_text: bool = True)``` **!!!**
# Colab
## training on colab you need to run next 3 cells or just use notebook (https://github.com/sovden/mountain_name_ner/blob/master/colab_training.ipynb):
1. run additional packages cell:
```
!pip install transformers[torch]
!pip install datasets
!pip install seqeval
```
2. run env preparation cell:
```
import shutil
import os

source_dir = '/content/mountain_name_ner/'
target_dir = '/content/'

if os.path.isdir(target_dir + "data_preparation"):
  for filename in os.listdir(target_dir):
      file_path = os.path.join(target_dir, filename)
      try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)
          elif os.path.isdir(file_path):
              shutil.rmtree(file_path)
      except Exception as e:
          print('Failed to delete %s. Reason: %s' % (file_path, e))

!git clone https://git_token@github.com/sovden/mountain_name_ner.git

file_names = os.listdir(source_dir)
    
for file_name in file_names:
    shutil.move(os.path.join(source_dir, file_name), target_dir)

os.rmdir(source_dir)
```
3. run training
```
from HF_training import run_training
```
## Example of simple model choosing:
It could be considered as some kind of Grid Search. Certainly, it should be rewrited with using some Bayesian Search or other optimization algorithm using **Optuna** package, for example:
```
for model_name in ["albert-base-v2", "bert-base-cased", "dslim/bert-base-NER", "Jorgeutd/albert-base-v2-finetuned-ner"]:
  model_iteration = 0
  for learning_rate in [3e-5, 5e-5, 8e-5]:
    for epoch_num in [1, 2]:
      save_name = f"{model_name}_{model_iteration}"
      run_training(is_colab=True,
                   save_name=save_name,
                   model_name=model_name,
                   num_train_epochs=epoch_num,
                   learning_rate=learning_rate,
                   dataset_params=(datasets, label_list))
      
      model_iteration += 1
```
## Inference
```
from inference_util import PrepareModel

start_model = PrepareModel(is_colab=True, is_lower_text=True)
start_model.predict_entity("our module landed at the foot of Arutaruki")
start_model.predict_entity("we tried to climb Hoverla where we saw top of Pip Ivan")
start_model.predict_entity("The scientist Marie Curie won the Nobel Prize in Physics and Chemistry.")
```
