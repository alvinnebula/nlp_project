# Glint Survey NLP

## Package Install Tips: (TBD)
```
pip install bertopic
```

It often encounters a time-out error on GCP. If this happens, simply re-run `pip install bertopic`. It will resume from its installation checkpoints.

- Installing Instruction 
  
## Data:

#### HCA Healthcare De-identified 2023 Glint Employee Surveys
_Comments are often misspelled, unstructured, multi-topics, packed with nuance, or in different languages which presents technical challenges for automatically processing into actionable insights​_

----
#### Debrief
- **Data Type**: Free-text
- **Collection Period**: 2023, Second-Round
- **Confidentiality**: De-identified

#### Content

1. **Manager**
2. **Organization**
3. **Personal**
4. **Resource**
5. **Team**

#### Key Statistics

- **Total Responses**: 186,016
- **Total Free-Text Comments**: 361,201

---

## Method / Pseudo Code:

<img width="2388" alt="new_diagram" src="https://github.com/alvinnebula/nlp_project/assets/89406404/6b89f33f-2c1b-4858-a35e-1c3714675108">

#### Pre-processing
_No NA Removal in this step_
```
src/basic/Preprocessor.py

------- # Load data by filter any specific feature columns, optional:
                                        'group',
                                        'division',
                                        'lob',
                                        'function',
                                        'subfunction',
                                        'processlevel',
                                        'job_class',
                                        'job_code',
                                        'rn_tenure'

-------- # Split data by 5 different comment contents as output attribute:
                    feature columns + ['Manager COMMENT']      -> output.manager_data,
                    feature columns + ['Organization COMMENT'] -> output.org_data
                    feature columns + ['Personal COMMENT']     -> output.personal_data
                    feature columns + ['Resources COMMENT']    -> output.resource_data
                    feature columns + ['Team COMMENT']         -> output.team_data
```
#### Semantic Split (Optional)
```
import vertexai.language_models
TextGenerationModel.from_pretrained('text-bison')
```
Comments will be splited by semantic difference
```
comment = "Our IVs have recently changed and they break off and are unreliable. Our ACE wraps are also unreliable and have no support. We need more medications in the pyxis."

output[0] = "Our IVs have recently changed and they break off and are unreliable."
output[1] = "Our ACE wraps are also unreliable and have no support."
output[2] = " We need more medications in the pyxis."
```

#### Categorization
_Core Function_

![BERTopic_Lego](https://github.com/alvinnebula/nlp_project/assets/89406404/8fa26161-9565-4eff-a603-3fdd6602c20d)

```
BERTopic: Highly flexible parameter & sub-models combination


topic_model = BERTopic(
                        top_n_words,
                        calculate_probabilities,
                        UMAP(...),
                        HDBSCAN(...),
                        CountVectorizer(...),
                        ClassTfidfTransformer(...),
                        KeyBERTInspired(...)    
                      )
```
BERTopic Parameters:
- `top_n_words`: The number of words per topic to extract.
- `calculate_probabilities`: Calculate the probabilities of all topics per document. Set to `True` for following hyperparameter tuning.

Sub-models:
- *Embedding Clustering*
    - `UMAP()`: A solution to __reduce dimensionality__ of the embeddings to a workable dimensional space for clustering algorithms to work with.
    - `HDBSCAN()`: To __cluster similar embeddings into groups__ to extract our topics, after reducing the dimensionality (get more accurate our topic representations).
- *Representation Refining*
    - `CountVectorizer()`: To __improve quality__ of topic representations on __phrase level__.
    - `ClassTfidfTransformer()`: To makes the documents in one cluster __different__ from other clusters.
- *Optional Finetuning*
    - `KeyBERTInspired()`: To fine-tune based on __semantic relationship__ between keywords/keyphrases.

```
src/basic/Categorizer.py

------- # Text Normalization by NLTK
--------------------- ## Cleaning
--------------------- ## Lemmatization

------- # NA Removal (ps,1)

------- # HyperParameter Modification by data length (ps,2)
--------------------- ## UMAP             n_neighbors, n_components
--------------------- ## HDBSCAN          min_cluster_size
--------------------- ## CountVectorizer  min_df

------- # Model Training
--------------------- ## Document to embeddings (Pretrained)
--------------------- ## Fit model on documents
--------------------- ## Generate topics and return the probabilities*

------- # Model Tuning (Temp Solutions)
--------------------- ## hard_tune()      Merging sub topic branches 
--------------------- ## soft_tune()      Reducing outlier by assigning topics to non-topic documents (-1) based on probabilities

------- # Representative:Probability Extraction 
--------------------- ## Representative   Obtaining highest topic contribution comment for each topic (for next step, reason we set calculate_probabilities = True)

-------- # generate 3 datasets for next step:
            Topic/ Comment Counts/ Representations               -> output.topic_data,
            For each topic, top 20 representative comments       -> output.document_data,
            Augment topic modeling results to original data     -> output.full_data
```
- `ps, 1`: for some meaningless comments could be converted to ['Nothing', 'Unknown', 'Missing', 'Null'] by Lemmatization
- `ps, 2`: BERTopic performs generall well for a length = (200 - 4000) pieces of comments

------------------------------------
By this line, topic categorization completed 

#### Summarization
_Core Function_
```
src/basic/Summarizer.py

------- # Preparing Input
--------------------- ## Comments Concatenation

------- # Generateing Group Names
--------------------- ## Prompt Engineering by representation_list, representative

------- # Generating Summaries
--------------------- ## Prompt Engineering by word:probability dictionary, representative

------- # Refining Group Names
--------------------- ## Prompt Engineering by group name, summary

-------- # generate a dataset as an augmentation to topic_data:
            Topic/ Comment Counts/ Summary               -> output.result_data
```

#### Topic Deep Dive
Very similar strategy to apply topic modeling & prompt engineering on a selected topic category. To explore deeper insight within the selected topic.

_Core Function_

Same core functions as the categorizer and summarizer, but with parameters optimized for smaller datasets.
```
src/basic/InnerAnalysis.py 

------- # Topic Modeling on selected topic
--------------------- ## Applying BERTopic on a smaller dataset

------- # Generateing Sub-Group Names & Summary
--------------------- ## Applying BERTopic on a smaller dataset

------- # Get original comments with each sub-topic
```

#### Excel Sheets
```
src/basic/ExcelGenerator.py

-------- # generate an excel file with two excel sheets:
      Sheet "Data": Augment prompt engineering results to original data
      Sheet "Detail": Topic/ Comment Counts/ Summary                          

```


