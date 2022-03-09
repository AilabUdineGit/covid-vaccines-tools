# covid-vaccines-tools

<p align="center">
    
**COVID-19 Vaccine Opinion Analysis** <br>
**Monitoring the Vaccines through Twitter Analysis** <br>
    
<img src="http://ailab.uniud.it/covid-vaccines/_nuxt/img/en-full-architecture.082a491.png" alt="system image" style="width:50%;"/>
</p>

Repository containing the code and data used for the interactive dashboard http://ailab.uniud.it/covid-vaccines/.



## Setup



### Environment setup

```
python3 -m pip install --user virtualenv
python3.6 -m venv env
source env/bin/activate

pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
python -m spacy download en

python
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('punkt')
>>> nltk.download('stopwords')
```

### Twitter API setup

You will need twitter credentials to use the twitter API.
They are saved in the file `CREDENTIALS.py`, which is not tracked (see `.gitignore`).
Create a file named `CREDENTIALS.py` containing the following variables.

```
CONSUMER_KEY = "..."
CONSUMER_SECRET = "..."
ACCESS_TOKEN = "..."
ACCESS_TOKEN_SECRET = "..."
BEARER_TOKEN = "..."
```

See the [twitter developer documentation](https://developer.twitter.com/en) for instructions on how to obtain them.

### Model setup

The pretrained model to predict the symptoms contained in the tweets is managed via [Git Large File Storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage) and can be found in:

```
model/ML4H_blind_smm4h_BERT_CRF/config.json
model/ML4H_blind_smm4h_BERT_CRF/pytorch_model.bin
```




## How to use
    


Our collected and processed data can be found in `data/yyyy-mm.csv` (one file for each month).
Data collection is still in progress and the data in the repository will be updated regularly.

_**The previous files `full_db.csv` and `full_db.pkl` contain the same data and will be removed shortly, as the monthly uploads are easier to maintain.**_

The repository also includes a subset of 1000 tweets which have been manually annotated for sentiment and adverse events.
The annotated data can be found in `data/subset_1000.csv` and `data/subset_1000.pkl`.

Format of the collected data
    
    +-----------------+----------------------------------------------------------------------------
    | column name     | content 
    +-----------------+----------------------------------------------------------------------------
    | tweet_id        | str, unique tweet identifier
    | created_at      | str, date of creation of the tweet, format "yyyy-mm-dd"
    | lang            | str, language of the tweet (example: "en", "es", "it", "fr")
    | country         | str, country code extracted by the Localization module
    |                 | (example: "US", "IT", "FR")
    | ent_hashtags    | list, hashtags contained in the tweet
    |                 | example [{"text": "Nepal", "indices": [137, 143]}]
    | ent_urls        | list, urls contained in the tweet 
    |                 | example [{
    |                 |           'url': 'https://t.co/O2FAJr7mL7',
    |                 |           'expanded_url': 'http://wrld.bg/ZB6O50HGXZI',
    |                 |           'display_url': 'wrld.bg/ZB6O50HGXZI',
    |                 |           'indices': [219, 242]
    |                 |         }]
    | vaccines        | tuple, tuple of names of the vaccines mentioned in the tweet
    | sentiment_value | dict, sentiment predicted based on the text of the tweet
    |                 | example {'label': 'LABEL_1', 'score': 0.8201785087585449}
    |                 | possible labels: LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
    |                 | score: confidence of the model on the predicted label
    | entities        | list, entities predicted by the model, including position in the text
    |                 | of the tweet, original text, normalized text and italian translation
    |                 | example [{
    |                 |           'position': {'start': 129.0, 'end': 137.0},
    |                 |           'text': {
    |                 |              'full_eng': 'myelitis',
    |                 |              'norm_eng': 'myelitis',
    |                 |              'full_ita': 'mielite'}
    |                 |          }]
    | user_id         | str, unique user identifier
    | ym              | str, year and month of creation of the tweet, format "yyyy-mm"
    +-----------------+----------------------------------------------------------------------------

Format of the annotated data

    +-----------------+----------------------------------------------------------------------------
    | column name     | content 
    +-----------------+----------------------------------------------------------------------------
    | user_name       | str, user name
    | sentiment_gold  | int, possible values: -1, 0, 1, sentiment determined by human annotators
    | sentiment_pred  | int, possible values: -1, 0, 1, sentiment determined by the sentiment module
    | entities_gold   | list of str, entities extracted by human annotators
    | entities_pred   | list of str, entities extracted by the symptom extraction module
    | tweet_id        | str, tweet identifier
    | user_id         | str, unique user identifier
    | created_at      | str, date of creation of the tweet, format "yyyy-mm-dd"
    | gold_int        | list of tuples, recording the character-level position of the strings
    |                 | in entities_gold. Tuples have the format (start_char, end_char, ent_type)
    |                 | example [(74, 92, 'ADE'), (224, 243, 'ADE')]
    | pred_int        | list of tuples, recording the character-level position of the strings in 
    |                 | in entities_pred. Tuples have the format (start_char, end_char, ent_type)
    |                 | example [(74, 92, 'ADE'), (224, 243, 'ADE')]
    | text            | str, text of the tweet
    |                 | the text has been redacted, repleacing all characters with "-" unless they
    |                 | are part of a gold/pred entity. This text can be used as is to reproduce
    |                 | the results in the paper, or can replaced with the actual text of the tweets
    +-----------------+----------------------------------------------------------------------------

All results and plots can be reproduced using the scripts in `plots`.


## How to run on your own



Run `full_pipeline.py` if you want to collect and analyze tweets on your own.

This will generate the following:

- logs/
  <br> Folder with logging files. They contain most of the output that can be seen on the terminal when executing the script.
- tmp_dataframes/
  <br> Folder for temporary files. Is automatically emptied at the end of a successful excecution. If execution is interrupted or crashes, the temporary files will be used to during the next run of `full_pipeline.py` to continue data processing without starting over.
- output_dataframe.pkl
  <br> Main output file. A pandas DataFrame containing the following columns: 
```
tweet_id
created_at
text_for_unique
lang
user_location_norm
country
ent_hashtags
ent_urls
vaccines
sentiment_value
entities
```
- entity_translation.pkl
  <br> Table to keep track of the entities extracted from the tweets (english) and their italian translation.
- locations.pkl
  <br> Table to keep track of the raw locations extracted from the tweets and their country.

- output_dataframe.csv
  <br> Human-readable version of output_dataframe.pkl in csv format. This file is never read by the script.
- output_dataframe.json
  <br> Human-readable version of output_dataframe.pkl in json format. This file is never read by the script.
- entity_translation.csv
  <br> Human-readable version of entity_translation.pkl in csv format. This file is never read by the script.
- locations.csv
  <br> Human-readable version of locations.pkl in csv format. This file is never read by the script.


