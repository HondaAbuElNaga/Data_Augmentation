<h1>make Data Augmentation of your questions </h1>
<h1>data must be in json format </h1>
<h2> ## Final Output

The script's final output is a single text file named **`sft.jsonl`**. This file is saved to the following path:
`C:\Users\mohan\OneDrive\Desktop\Augmentation\data_dir/sft.jsonl`

This file contains the **Augmented Data** for training. For every original question-answer (Q&A) pair found in the `qa_data_from_pdf` list, the script generates 4 new, diverse rephrasings of the original question.

The final `sft.jsonl` file will contain 5 lines for each original Q&A pair (1 original question + 4 new variations). Each line is formatted as a JSON record, making it ready for the Supervised Fine-Tuning (SFT) process.
<h2/>

<h2>Prerequisites</h2>
<ul>
  <li>Python 3.11+</li>
</ul>

<h2>Installation</h2>
<h3>1. Clone the repository:</h3>

```
git clone https://github.com/ThomasJanssen-tech/Chatbot-with-RAG-and-LangChain.git
cd Chatbot-with-RAG-and-LangChain
```

<h3>2. Create a virtual environment</h3>

```
python -m venv venv
```

<h3>3. Activate the virtual environment</h3>

```
venv\Scripts\Activate
(or on Mac): source venv/bin/activate
```

<h3>4. Install libraries</h3>

```
pip install -r requirements.txt
```

<h3>5. Add OpenAI API Key</h3>
Rename the .env.example file to .env
Add your OpenAI API Key

<h2>Executing the scripts</h2>

- Open a terminal in VS Code

- Execute the following command:

```
Data_Augmentation.py
```
