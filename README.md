# covid-vaccines-tools

<p align="center">
    
**COVID-19 Vaccine Opinion Analysis** <br>
**Monitoring the Vaccines through Twitter Analysis** <br>
    
<img src="http://ailab.uniud.it/covid-vaccines/_nuxt/img/en-full-architecture.082a491.png" alt="system image" style="width:50%;"/>
</p>

Repository containing the code and data used for the interactive dashboard http://ailab.uniud.it/covid-vaccines/.

<details><summary>

## Setup

</summary>

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
</details>

<details><summary>

## How to use
    
</summary>

Our collected and processed data can be found in `full_db.pkl` (and in the human-readable copy `full_db.csv`).
Data collection is still in progress and the data in the repository will be updated regularly.

Format
    
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
    +-----------------+----------------------------------------------------------------------------

    
</details>

<details><summary>

## How to run on your own

</summary>

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

</details>
