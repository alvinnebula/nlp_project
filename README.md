# Glint Survey NLP

## Package Requirements: (TBD)
```
pip install bertopic
```

It often encounters a time-out error on GCP. If this happens, simply re-run `pip install bertopic`. It will resume from its installation checkpoints.

> - Installing Instruction 
  
## Data:

#### HCA Healthcare De-identified Glint Employee Survey 2023
_Comments are often misspelled, unstructured, multi-topics, packed with nuance, or in different languages which presents technical challenges for automatically processing into actionable insights​_

----
#### Debriefs
> - **Data Type**: Free-text
> - **Collection Period**: 2023, Second-Round
> - **Confidentiality**: De-identified

#### Contents

> 1. **Manager**
> 2. **Organization**
> 3. **Personal**
> 4. **Resource**
> 5. **Team**

#### Key Statistics

> - **Total Responses**: 186,016
> - **Total Free-Text Comments**: 361,201

---
## Data Privacy: (TBD)
_HR TEAM EXCEL FILE​, De-id level: remove employee demographics_

## Method / Pseudo Code:
<img width="2388" alt="new_diagram" src="https://github.com/HCA-CTI/glint_survey_nlp/assets/143527820/ece79975-cbce-4c79-a3db-e1877434b871">

---

#### Pre-processing
> _No NA Removal in this step_

- *Code Breakdown*
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

---

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
---

#### Categorization

> _Core Function_

<table>
  <tr>
    <!-- Image Column -->
    <td>
      <img src="https://github.com/HCA-CTI/glint_survey_nlp/assets/143527820/c09c8225-94f7-461a-8129-f5bfd1d06d98" alt="BERTopic_Lego">
    </td>
    <!-- Code Column -->
    <td>
      <pre>
BERTopic: Highly flexible parameter & sub-models combination
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
topic_model = BERTopic(
                        &nbsp;&nbsp; top_n_words,
                        &nbsp;&nbsp; calculate_probabilities,
                        &nbsp;&nbsp;
                        &nbsp;&nbsp; KeyBERTInspired(...),
                        &nbsp;&nbsp; ClassTfidfTransformer(...),
                        &nbsp;&nbsp; CountVectorizer(...),
                        &nbsp;&nbsp; HDBSCAN(...),
                        &nbsp;&nbsp; UMAP(...) 
                      &nbsp; )
      </pre>
    </td>
  </tr>
</table>

**BERTopic Parameters:**
 - `top_n_words`: The number of words per topic to extract.
 - `calculate_probabilities`: Calculate the probabilities of all topics per document. Set to `True` for following hyperparameter tuning.

**Sub-models:**
- *Embedding Clustering*
    - `UMAP()`: A solution to __reduce dimensionality__ of the embeddings to a workable dimensional space for clustering algorithms to work with.
    - `HDBSCAN()`: To __cluster similar embeddings into groups__ to extract our topics, after reducing the dimensionality (get more accurate our topic representations).
- *Representation Refining*
    - `CountVectorizer()`: To __improve quality__ of topic representations on __phrase level__.
    - `ClassTfidfTransformer()`: To makes the documents in one cluster __different__ from other clusters.
- *Optional Finetuning*
    - `KeyBERTInspired()`: To fine-tune based on __semantic relationship__ between keywords/keyphrases.

- *Code Breakdown*
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
            'Topic'/ 'Comment Counts'/ 'Representations'        -> output.topic_data,
            For each topic, top 20 representative comments      -> output.document_data,
            Augment topic modeling results to original data     -> output.full_data
```
- `ps, 1`: for some meaningless comments could be converted to ['Nothing', 'Unknown', 'Missing', 'Null'] by Lemmatization
- `ps, 2`: BERTopic performs generall well for a length = (200 - 4000) pieces of comments

------------------------------------
By this line, topic categorization completed 

---

#### Summarization
> _Core Function_

- *Code Breakdown*
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
                      'Topic'/ 'Comment Counts'/ 'Summary'  ->  output.result_data
```
---

#### Topic Deep Dive
Very similar strategy to apply topic modeling & prompt engineering on a selected topic category. To explore deeper insight within the selected topic.

> _Core Function_

- *Code Breakdown*
Same core functions as the categorizer and summarizer, but with parameters optimized for smaller datasets.
```
src/basic/InnerAnalysis.py 

------- # Topic Modeling on selected topic
--------------------- ## Applying BERTopic on a smaller dataset

------- # Generateing Sub-Group Names & Summary
--------------------- ## Applying BERTopic on a smaller dataset

------- # Get original comments with each sub-topic
```
---

#### Excel Sheets
- *Code Breakdown*
```
src/basic/ExcelGenerator.py

-------- # generate an excel file with two excel sheets:
                          Sheet "Data": Augment 'Topic' to original data
                          Sheet "Detail": 'Topic'/ 'Comment Counts'/ 'Summary'                          

```
## LLM Hallucination / Omission: (TBD)

